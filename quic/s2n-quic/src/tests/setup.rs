// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    client::Connect,
    provider::{
        event,
        io::testing::{primary, spawn, Handle, Result},
    },
    stream::PeerStream,
    Client, Server,
};
use rand::{Rng, RngCore};
use s2n_quic_core::{crypto::tls::testing::certificates, havoc, stream::testing::Data};
use std::net::SocketAddr;
use once_cell::sync::Lazy;
use std::sync::{Arc, Mutex};
use std::io::Write;
use tracing_subscriber::fmt::writer::TestWriter;
use aws_sdk_bedrockruntime::{
    Client as BedrockClient,
    Error,
    types::{Message, ConversationRole, ContentBlock, InferenceConfiguration},
};
use bach::time::scheduler::{self, Scheduler};

pub static SERVER_CERTS: (&str, &str) = (certificates::CERT_PEM, certificates::KEY_PEM);

pub static LOG_BUFFER: Lazy<Arc<Mutex<Vec<u8>>>> = Lazy::new(|| Arc::new(Mutex::new(Vec::new())));

/// Writes into our LOG_BUFFER
struct BufferWriter(Arc<Mutex<Vec<u8>>>);
impl Write for BufferWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let mut buf = self.0.lock().unwrap();
        buf.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

/// Fan-out writer: first W1, then W2
struct MultiWriter<W1, W2>(W1, W2);
impl<W1: Write, W2: Write> Write for MultiWriter<W1, W2> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.0.write(buf)?;
        self.1.write_all(buf)?;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.flush().and_then(|_| self.1.flush())
    }
}

pub fn tracing_events() -> event::tracing::Subscriber {
    use std::sync::Once;

    static TRACING: Once = Once::new();

    // make sure this only gets initialized once
    TRACING.call_once(|| {
        let format = tracing_subscriber::fmt::format()
            .with_level(false) // don't include levels in formatted output
            .with_timer(Uptime)
            .with_ansi(false)
            .compact(); // Use a less verbose output format.

        struct Uptime;

        // Generate the timestamp from the testing IO provider rather than wall clock.
        impl tracing_subscriber::fmt::time::FormatTime for Uptime {
            fn format_time(
                &self,
                w: &mut tracing_subscriber::fmt::format::Writer<'_>,
            ) -> std::fmt::Result {
                write!(w, "{}", crate::provider::io::testing::now())
            }
        }

        let env_filter = tracing_subscriber::EnvFilter::builder()
            .with_default_directive(tracing::Level::DEBUG.into())
            .with_env_var("S2N_LOG")
            .from_env()
            .unwrap();

        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .event_format(format)
            .with_writer(|| {
                // TestWriter prints to the console exactly as before…
                let console = TestWriter::new();
                // …and BufferWriter grabs every byte into LOG_BUFFER.
                let buffer = BufferWriter(LOG_BUFFER.clone());
                MultiWriter(console, buffer)
            })
            .init();
    });

    event::tracing::Subscriber::default()
}

pub fn start_server(mut server: Server) -> Result<SocketAddr> {
    let server_addr = server.local_addr()?;

    // accept connections and echo back
    spawn(async move {
        while let Some(mut connection) = server.accept().await {
            tracing::debug!("accepted server connection: {}", connection.id());
            spawn(async move {
                while let Ok(Some(stream)) = connection.accept().await {
                    tracing::debug!("accepted server stream: {}", stream.id());
                    match stream {
                        PeerStream::Receive(mut stream) => {
                            spawn(async move {
                                while let Ok(Some(_)) = stream.receive().await {
                                    // noop
                                }
                            });
                        }
                        PeerStream::Bidirectional(mut stream) => {
                            spawn(async move {
                                while let Ok(Some(chunk)) = stream.receive().await {
                                    let _ = stream.send(chunk).await;
                                }
                            });
                        }
                    }
                }
            });
        }
    });

    Ok(server_addr)
}

pub fn server(handle: &Handle) -> Result<SocketAddr> {
    let server = build_server(handle)?;
    start_server(server)
}

pub fn build_server(handle: &Handle) -> Result<Server> {
    Ok(Server::builder()
        .with_io(handle.builder().build().unwrap())?
        .with_tls(SERVER_CERTS)?
        .with_event(tracing_events())?
        .with_random(Random::with_seed(123))?
        .start()?)
}

pub fn client(handle: &Handle, server_addr: SocketAddr) -> Result {
    let client = build_client(handle)?;
    start_client(client, server_addr, Data::new(10_000))
}

/// Send a single prompt to Bedrock via the Conversation API and block for the reply.
pub fn send_to_bedrock_sync(prompt: &str) -> Result<String, Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async move {
        // 1. Load config (avoid the deprecated `from_env` helper).
        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region("us-east-1")
            .load()
            .await;
        let client = BedrockClient::new(&config);

        // 2. Build the *user* message.
        //
        // `ContentBlock` is a *union* (an enum in Rust) – it has no
        // `builder()` method.  Use the `Text` variant directly.
        let user_msg = Message::builder()
            .role(ConversationRole::User)
            .content(ContentBlock::Text(prompt.to_owned()))
            .build()?;          // `build()` → Result<_, BuildError>

        // 3. Inference parameters.
        let inf_cfg = InferenceConfiguration::builder()
            .temperature(0.8)
            .top_p(0.9)
            .build();

        // 4. Call the model.
        let resp = client
            .converse()
            .model_id("us.meta.llama4-scout-17b-instruct-v1:0")
            .messages(user_msg)
            .inference_config(inf_cfg)
            .send()
            .await?;

        // 5. Pull the first text chunk out of the reply:
        //    outer `output()` → Option<&types::ConverseOutput>
        //    inner `as_message()` unwraps the `Message` variant
        //    finally scan the content blocks for the first `Text` field.
        let reply_text = resp
            .output()
            .and_then(|o| o.as_message().ok())          // `ConverseOutput::Message`
            .and_then(|m| {
                m.content().iter().find_map(|c| {
                    if let ContentBlock::Text(t) = c { Some(t.clone()) } else { None }
                })
            })
            .unwrap_or_default();

        Ok(reply_text)
    })
}

fn init_scheduler() -> Option<scheduler::Handle> {
    scheduler::scope::set(Some(Scheduler::new().handle()))
}

/// Grab everything from the in-memory logger and return a UTF-8 String.
pub fn collect_build_logs() -> String {
    let data = LOG_BUFFER.lock().unwrap().clone();
    String::from_utf8_lossy(&data).into_owned()
}

/// Build the Bedrock prompt around the logs and send for analysis.
pub fn analyze_build_logs() -> Result<String, Error> {
    let _restore = init_scheduler();
    let all_logs = collect_build_logs();

    let prompt = format!(
        r#"You are an expert Rust build and log‐analysis assistant.

        ## Brief Summary  
        Provide exactly two sentences summarizing what happened.

        ## Errors, Warnings, and Likely Causes  
        List any errors or warnings (with line numbers if available) and your best guess at their causes.  
        If none are found, omit this section entirely.

        ## Anomalies or Performance Observations  
        Highlight any unusual timings, retransmissions, stalls, or patterns that could indicate inefficiencies.  
        If there’s nothing notable, skip this section.

        ## Conclusion  
        If everything is clean, respond with exactly one concise sentence:  
        “No issues detected. Build succeeded cleanly.”  
        Otherwise, summarize in one sentence.

        —BEGIN LOGS—  
        {}  
        —END LOGS—  
        "#,
        all_logs
    );

    send_to_bedrock_sync(&prompt)
}

pub fn start_client(client: Client, server_addr: SocketAddr, data: Data) -> Result {
    primary::spawn(async move {
        let connect = Connect::new(server_addr).with_server_name("localhost");
        let mut connection = client.connect(connect).await.unwrap();

        tracing::debug!("connected with client connection: {}", connection.id());

        let stream = connection.open_bidirectional_stream().await.unwrap();
        tracing::debug!("opened client stream: {}", stream.id());

        let (mut recv, mut send) = stream.split();

        let mut send_data = data;
        let mut recv_data = data;

        primary::spawn(async move {
            while let Some(chunk) = recv.receive().await.unwrap() {
                recv_data.receive(&[chunk]);
            }
            assert!(recv_data.is_finished());
        });

        while let Some(chunk) = send_data.send_one(usize::MAX) {
            tracing::debug!("client sending {} chunk", chunk.len());
            send.send(chunk).await.unwrap();
        }
    });

    Ok(())
}

pub fn build_client(handle: &Handle) -> Result<Client> {
    Ok(Client::builder()
        .with_io(handle.builder().build().unwrap())?
        .with_tls(certificates::CERT_PEM)?
        .with_event(tracing_events())?
        .with_random(Random::with_seed(123))?
        .start()?)
}

pub fn client_server(handle: &Handle) -> Result<SocketAddr> {
    let addr = server(handle)?;
    client(handle, addr)?;
    Ok(addr)
}

pub struct Random {
    inner: rand_chacha::ChaCha8Rng,
}

impl Random {
    pub fn with_seed(seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            inner: rand_chacha::ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl havoc::Random for Random {
    fn fill(&mut self, bytes: &mut [u8]) {
        self.fill_bytes(bytes);
    }

    fn gen_range(&mut self, range: std::ops::Range<u64>) -> u64 {
        self.inner.random_range(range)
    }
}

impl RngCore for Random {
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest)
    }

    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }
}

impl crate::provider::random::Provider for Random {
    type Generator = Self;

    type Error = core::convert::Infallible;

    fn start(self) -> Result<Self::Generator, Self::Error> {
        Ok(self)
    }
}

impl crate::provider::random::Generator for Random {
    fn public_random_fill(&mut self, dest: &mut [u8]) {
        self.fill_bytes(dest);
    }

    fn private_random_fill(&mut self, dest: &mut [u8]) {
        self.fill_bytes(dest);
    }
}

#[cfg(not(target_os = "windows"))]
mod mtls {
    use super::*;
    use crate::provider::tls;

    pub fn build_client_mtls_provider(ca_cert: &str) -> Result<tls::default::Client> {
        let tls = tls::default::Client::builder()
            .with_certificate(ca_cert)?
            .with_client_identity(
                certificates::MTLS_CLIENT_CERT,
                certificates::MTLS_CLIENT_KEY,
            )?
            .build()?;
        Ok(tls)
    }

    pub fn build_server_mtls_provider(ca_cert: &str) -> Result<tls::default::Server> {
        let tls = tls::default::Server::builder()
            .with_certificate(
                certificates::MTLS_SERVER_CERT,
                certificates::MTLS_SERVER_KEY,
            )?
            .with_client_authentication()?
            .with_trusted_certificate(ca_cert)?
            .build()?;
        Ok(tls)
    }
}

mod slow_tls {
    use crate::provider::tls::Provider;
    use s2n_quic_core::crypto::tls::{slow_tls::SlowEndpoint, Endpoint};
    pub struct SlowTlsProvider<E: Endpoint> {
        pub endpoint: E,
    }

    impl<E: Endpoint> Provider for SlowTlsProvider<E> {
        type Server = SlowEndpoint<E>;
        type Client = SlowEndpoint<E>;
        type Error = String;

        fn start_server(self) -> Result<Self::Server, Self::Error> {
            Ok(SlowEndpoint::new(self.endpoint))
        }

        fn start_client(self) -> Result<Self::Client, Self::Error> {
            Ok(SlowEndpoint::new(self.endpoint))
        }
    }
}

#[cfg(feature = "s2n-quic-tls")]
mod resumption {
    use super::*;
    use crate::provider::tls::{
        self,
        s2n_tls::{
            callbacks::{ConnectionFuture, SessionTicket, SessionTicketCallback},
            config::ConnectionInitializer,
            connection::Connection,
            error::Error,
            Server,
        },
    };
    use std::{
        collections::VecDeque,
        pin::Pin,
        sync::{Arc, Mutex},
    };

    pub static TICKET_KEY: [u8; 16] = [0; 16];
    #[derive(Default, Clone)]
    pub struct SessionTicketHandler {
        ticket_storage: Arc<Mutex<VecDeque<Vec<u8>>>>,
    }

    impl SessionTicketCallback for SessionTicketHandler {
        fn on_session_ticket(&self, _connection: &mut Connection, session_ticket: &SessionTicket) {
            let size = session_ticket.len().unwrap();
            let mut data = vec![0; size];
            session_ticket.data(&mut data).unwrap();
            let mut vec = (*self.ticket_storage).lock().unwrap();
            vec.push_back(data);
        }
    }

    impl ConnectionInitializer for SessionTicketHandler {
        fn initialize_connection(
            &self,
            connection: &mut Connection,
        ) -> Result<Option<Pin<Box<(dyn ConnectionFuture)>>>, Error> {
            if let Some(ticket) = (*self.ticket_storage).lock().unwrap().pop_back().as_deref() {
                connection.set_session_ticket(ticket)?;
            }
            Ok(None)
        }
    }

    pub fn build_server_resumption_provider(
        cert: &str,
        key: &str,
    ) -> Result<tls::default::Server<s2n_quic_tls_default::Server>> {
        let mut tls = Server::builder().with_certificate(cert, key)?;

        let config = tls.config_mut();
        config.enable_session_tickets(true)?;
        config.add_session_ticket_key(
            "keyname".as_bytes(),
            &TICKET_KEY,
            std::time::SystemTime::now(),
        )?;

        let tls = Server::from_loader(tls.build()?);
        Ok(tls)
    }

    pub fn build_client_resumption_provider(
        cert: &str,
        handler: &SessionTicketHandler,
    ) -> Result<tls::default::Client> {
        let mut tls = tls::s2n_tls::Client::builder().with_certificate(cert)?;
        let config = tls.config_mut();
        config
            .enable_session_tickets(true)?
            .set_session_ticket_callback(handler.clone())?
            .set_connection_initializer(handler.clone())?;
        Ok(tls.build()?)
    }
}

#[cfg(not(target_os = "windows"))]
pub use mtls::*;

#[cfg(feature = "s2n-quic-tls")]
pub use resumption::*;

pub use slow_tls::SlowTlsProvider;
