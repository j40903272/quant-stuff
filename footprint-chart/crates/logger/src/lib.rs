use std::fs::OpenOptions;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub fn init_debug_logger() {
    let debug_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("log-debug.log")
        .unwrap();

    let subscriber = tracing_subscriber::fmt::layer()
        // .with_file(true)
        // .with_line_number(true)
        .with_thread_ids(true);
    // .with_thread_names(true)

    let file_subscriber = tracing_subscriber::fmt::layer()
        // .with_file(true)
        // .with_line_number(true)
        .with_thread_ids(true)
        // .with_thread_names(true)
        .with_ansi(false)
        .with_writer(debug_file);

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(subscriber)
        .with(file_subscriber)
        .init();
    tracing::info!("debug logger initialized");
}

pub fn init_debug_logger_with_file(filepath: &str) {
    let debug_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(filepath)
        .unwrap();

    let subscriber = tracing_subscriber::fmt::layer()
        // .with_file(true)
        // .with_line_number(true)
        .with_thread_ids(true);
    // .with_thread_names(true)

    let file_subscriber = tracing_subscriber::fmt::layer()
        // .with_file(true)
        // .with_line_number(true)
        .with_thread_ids(true)
        // .with_thread_names(true)
        .with_ansi(false)
        .with_writer(debug_file);

    tracing_subscriber::registry()
        .with(EnvFilter::from_default_env())
        .with(subscriber)
        .with(file_subscriber)
        .init();
    // tracing::info!("debug logger initialized");
}
