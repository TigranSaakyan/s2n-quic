// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::tests::setup::LOG_BUFFER;
use bach::time::scheduler::{self, Scheduler};

use aws_sdk_bedrockruntime::{
    Client as BedrockClient,
    Error,
    types::{Message, ConversationRole, ContentBlock, InferenceConfiguration},
};

use super::*;
use crate::provider::tls::default::{self as tls, security};

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

fn test_policy(policy: &security::Policy) {
    let model = Model::default();

    test(model, |handle| {
        let server = tls::Server::from_loader({
            let mut builder = tls::config::Config::builder();
            builder
                .enable_quic()?
                .set_application_protocol_preference(["h3"])?
                .set_security_policy(policy)?
                .load_pem(
                    certificates::CERT_PEM.as_bytes(),
                    certificates::KEY_PEM.as_bytes(),
                )?;

            builder.build()?
        });

        let server = Server::builder()
            .with_io(handle.builder().build()?)?
            .with_tls(server)?
            .with_event(tracing_events())?
            .with_random(Random::with_seed(456))?
            .start()?;

        let client = tls::Client::from_loader({
            let mut builder = tls::config::Config::builder();
            builder
                .enable_quic()?
                .set_application_protocol_preference(["h3"])?
                .set_security_policy(policy)?
                .trust_pem(certificates::CERT_PEM.as_bytes())?;

            builder.build()?
        });

        let client = Client::builder()
            .with_io(handle.builder().build()?)?
            .with_tls(client)?
            .with_event(tracing_events())?
            .with_random(Random::with_seed(456))?
            .start()?;

        let addr = start_server(server)?;
        start_client(client, addr, Data::new(1000))?;
        Ok(addr)
    })
    .unwrap();
}

#[test]
fn default_fips_test() {
    let _restore = scheduler::scope::set(Some(Scheduler::new().handle()));

    // TODO switch this to `default_fips` when the policy supports TLS 1.3
    //      see https://github.com/aws/s2n-quic/issues/2247
    test_policy(&security::Policy::from_version("20230317").unwrap());

    let data = LOG_BUFFER.lock().unwrap().clone();
    let all_logs = String::from_utf8_lossy(&data);
    println!("\n=== DUPLICATED LOG DUMP ===\n{}", all_logs);

    let reply = send_to_bedrock_sync("hello! how are you? what is 1 + 3?")
        .expect("bedrock call failed");
    println!("Bedrock replied: {}", reply);
}
