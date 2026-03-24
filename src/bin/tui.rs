//! ExpertFlow TUI — Real-time terminal dashboard for MoE expert streaming.
//!
//! Displays system overview, expert cache status, scheduler activity,
//! token generation metrics, and model information in a professional
//! terminal interface.

use std::io;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    prelude::*,
    widgets::*,
};
use rand::Rng;

use expertflow::model::config::MoEConfig;

// ─── Simulated metrics ───────────────────────────────────────────────────────

/// Expert thermal state
#[derive(Debug, Clone, Copy, PartialEq)]
enum ThermalState {
    Hot,
    Warm,
    Cold,
    Evicting,
}

impl ThermalState {
    fn color(&self) -> Color {
        match self {
            ThermalState::Hot => Color::Green,
            ThermalState::Warm => Color::Yellow,
            ThermalState::Cold => Color::DarkGray,
            ThermalState::Evicting => Color::Red,
        }
    }

    fn label(&self) -> &str {
        match self {
            ThermalState::Hot => "HOT",
            ThermalState::Warm => "WARM",
            ThermalState::Cold => "COLD",
            ThermalState::Evicting => "EVICT",
        }
    }
}

/// Simulated expert entry for the cache panel
#[derive(Debug, Clone)]
struct ExpertEntry {
    id: usize,
    layer: usize,
    state: ThermalState,
    temperature: f64,
    size_mb: f64,
    last_access_ms: u64,
}

/// Simulated scheduler task
#[derive(Debug, Clone)]
struct SchedulerTask {
    id: usize,
    kind: String,
    target: String,
    progress: f64,
}

/// Log entry for the scheduler decisions panel
#[derive(Debug, Clone)]
struct LogEntry {
    timestamp: String,
    level: LogLevel,
    message: String,
}

#[derive(Debug, Clone, Copy)]
enum LogLevel {
    Info,
    Warn,
    Debug,
}

impl LogLevel {
    fn color(&self) -> Color {
        match self {
            LogLevel::Info => Color::Cyan,
            LogLevel::Warn => Color::Yellow,
            LogLevel::Debug => Color::DarkGray,
        }
    }

    fn label(&self) -> &str {
        match self {
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Debug => "DBUG",
        }
    }
}

/// All dashboard state
struct DashboardState {
    // System
    total_ram_gb: f64,
    used_ram_gb: f64,
    gpu_util: f64,
    ane_util: f64,
    cpu_util: f64,
    nvme_read_gbs: f64,
    nvme_write_gbs: f64,
    memory_pressure: f64,
    kv_cache_gb: f64,
    expert_budget_gb: f64,

    // Expert cache
    experts: Vec<ExpertEntry>,
    cache_hits: u64,
    cache_misses: u64,
    evictions: u64,

    // Scheduler
    tasks: Vec<SchedulerTask>,
    logs: Vec<LogEntry>,

    // Token generation
    tokens_per_sec: f64,
    latency_ms: f64,
    tokens_generated: u64,
    prompt_tokens: u64,
    current_text: String,

    // Model
    config: MoEConfig,

    // UI state
    active_panel: usize,
    panel_count: usize,
    tick: u64,
    start_time: Instant,
}

impl DashboardState {
    fn new() -> Self {
        let config = MoEConfig::deepseek_v3_q4();

        let mut state = Self {
            total_ram_gb: 128.0,
            used_ram_gb: 78.4,
            gpu_util: 0.0,
            ane_util: 0.0,
            cpu_util: 0.0,
            nvme_read_gbs: 0.0,
            nvme_write_gbs: 0.0,
            memory_pressure: 0.0,
            kv_cache_gb: 0.0,
            expert_budget_gb: 0.0,

            experts: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
            evictions: 0,

            tasks: Vec::new(),
            logs: Vec::new(),

            tokens_per_sec: 0.0,
            latency_ms: 0.0,
            tokens_generated: 0,
            prompt_tokens: 0,
            current_text: String::new(),

            config,

            active_panel: 0,
            panel_count: 5,
            tick: 0,
            start_time: Instant::now(),
        };

        state.seed_initial_data();
        state
    }

    fn seed_initial_data(&mut self) {
        let mut rng = rand::thread_rng();

        // Seed some experts in various states
        let states = [
            ThermalState::Hot,
            ThermalState::Hot,
            ThermalState::Hot,
            ThermalState::Warm,
            ThermalState::Warm,
            ThermalState::Cold,
            ThermalState::Cold,
            ThermalState::Cold,
        ];

        for i in 0..24 {
            let state = states[i % states.len()];
            self.experts.push(ExpertEntry {
                id: rng.gen_range(0..256),
                layer: rng.gen_range(0..61),
                state,
                temperature: match state {
                    ThermalState::Hot => rng.gen_range(5.0..10.0),
                    ThermalState::Warm => rng.gen_range(1.0..5.0),
                    ThermalState::Cold => rng.gen_range(0.0..1.0),
                    ThermalState::Evicting => 0.0,
                },
                size_mb: rng.gen_range(1200.0..1600.0),
                last_access_ms: match state {
                    ThermalState::Hot => rng.gen_range(0..100),
                    ThermalState::Warm => rng.gen_range(100..2000),
                    ThermalState::Cold => rng.gen_range(2000..10000),
                    ThermalState::Evicting => rng.gen_range(10000..30000),
                },
            });
        }

        // Sort experts: hot first, then warm, cold, evicting
        self.experts.sort_by(|a, b| {
            b.temperature.partial_cmp(&a.temperature).unwrap()
        });

        // Initial logs
        self.logs = vec![
            LogEntry { timestamp: "00:00.000".into(), level: LogLevel::Info, message: "ExpertFlow scheduler initialized".into() },
            LogEntry { timestamp: "00:00.012".into(), level: LogLevel::Info, message: format!("Model: {} ({} experts, {} active/token)", self.config.name, self.config.num_experts, self.config.num_active_experts) },
            LogEntry { timestamp: "00:00.015".into(), level: LogLevel::Info, message: "Memory budget: 102.4 GB (128 GB total, 20% reserve)".into() },
            LogEntry { timestamp: "00:00.018".into(), level: LogLevel::Debug, message: "Pre-warming 10 hot experts from disk cache".into() },
            LogEntry { timestamp: "00:00.142".into(), level: LogLevel::Info, message: "Pre-warm complete, 10 experts loaded".into() },
            LogEntry { timestamp: "00:00.145".into(), level: LogLevel::Info, message: "Scheduler ready, waiting for inference requests".into() },
        ];

        // Seed initial tasks
        self.tasks = vec![
            SchedulerTask { id: 1, kind: "GPU".into(), target: "Expert 42 (L12)".into(), progress: 0.7 },
            SchedulerTask { id: 2, kind: "ANE".into(), target: "Expert 17 (L12)".into(), progress: 0.5 },
            SchedulerTask { id: 3, kind: "NVMe→RAM".into(), target: "Expert 191 (L13)".into(), progress: 0.3 },
        ];

        self.cache_hits = 1847;
        self.cache_misses = 203;
        self.evictions = 89;
        self.tokens_generated = 342;
        self.prompt_tokens = 128;
        self.tokens_per_sec = 18.7;
        self.latency_ms = 53.5;
        self.gpu_util = 0.82;
        self.ane_util = 0.45;
        self.cpu_util = 0.23;
        self.nvme_read_gbs = 5.2;
        self.nvme_write_gbs = 0.3;
        self.memory_pressure = 0.61;
        self.kv_cache_gb = 4.2;
        self.expert_budget_gb = 98.2;
        self.current_text = "The key insight behind Mixture of Experts is that not all parameters need to be active for every input token. By routing each token to only a subset of specialized experts".into();
    }

    fn tick_update(&mut self) {
        self.tick += 1;
        let mut rng = rand::thread_rng();

        // Smoothly vary metrics
        self.gpu_util = (self.gpu_util + rng.gen_range(-0.05..0.05)).clamp(0.4, 0.98);
        self.ane_util = (self.ane_util + rng.gen_range(-0.03..0.03)).clamp(0.1, 0.7);
        self.cpu_util = (self.cpu_util + rng.gen_range(-0.02..0.02)).clamp(0.05, 0.4);
        self.nvme_read_gbs = (self.nvme_read_gbs + rng.gen_range(-0.5..0.5)).clamp(0.0, 7.4);
        self.nvme_write_gbs = (self.nvme_write_gbs + rng.gen_range(-0.1..0.1)).clamp(0.0, 2.0);
        self.memory_pressure = (self.memory_pressure + rng.gen_range(-0.02..0.02)).clamp(0.3, 0.85);
        self.used_ram_gb = self.total_ram_gb * self.memory_pressure;
        self.tokens_per_sec = (self.tokens_per_sec + rng.gen_range(-1.5..1.5)).clamp(8.0, 32.0);
        self.latency_ms = 1000.0 / self.tokens_per_sec;
        self.kv_cache_gb = (self.kv_cache_gb + rng.gen_range(-0.1..0.15)).clamp(1.0, 12.0);
        self.expert_budget_gb = self.total_ram_gb * 0.8 - self.kv_cache_gb;

        // Simulate token generation
        if self.tick % 3 == 0 {
            self.tokens_generated += 1;
            let words = [", ", "the ", "model ", "can ", "achieve ", "significant ",
                        "efficiency ", "gains ", "while ", "maintaining ", "quality. ",
                        "Each ", "expert ", "specializes ", "in ", "different ",
                        "aspects ", "of ", "language ", "understanding "];
            let word = words[self.tokens_generated as usize % words.len()];
            self.current_text.push_str(word);
            if self.current_text.len() > 300 {
                self.current_text = self.current_text[self.current_text.len()-250..].to_string();
            }
        }

        // Occasionally update expert states
        if self.tick % 5 == 0 && !self.experts.is_empty() {
            let idx = rng.gen_range(0..self.experts.len());
            let new_temp = (self.experts[idx].temperature + rng.gen_range(-0.5..0.8)).max(0.0);
            self.experts[idx].temperature = new_temp;
            self.experts[idx].state = if new_temp > 5.0 {
                ThermalState::Hot
            } else if new_temp > 1.5 {
                ThermalState::Warm
            } else if new_temp > 0.2 {
                ThermalState::Cold
            } else {
                ThermalState::Evicting
            };
            self.experts[idx].last_access_ms = if self.experts[idx].state == ThermalState::Hot {
                rng.gen_range(0..100)
            } else {
                rng.gen_range(100..15000)
            };
        }

        // Sort experts by temperature
        self.experts.sort_by(|a, b| b.temperature.partial_cmp(&a.temperature).unwrap());

        // Cache stats
        if self.tick % 2 == 0 {
            if rng.gen_bool(0.85) {
                self.cache_hits += 1;
            } else {
                self.cache_misses += 1;
            }
        }
        if self.tick % 20 == 0 {
            self.evictions += 1;
        }

        // Update tasks
        for task in &mut self.tasks {
            task.progress = (task.progress + rng.gen_range(0.02..0.08)).min(1.0);
        }
        // Replace completed tasks
        self.tasks.retain(|t| t.progress < 1.0);
        while self.tasks.len() < 3 {
            let kinds = ["GPU", "ANE", "CPU", "NVMe→RAM"];
            let kind = kinds[rng.gen_range(0..kinds.len())];
            let expert_id = rng.gen_range(0..256);
            let layer = rng.gen_range(0..61);
            self.tasks.push(SchedulerTask {
                id: self.tick as usize,
                kind: kind.into(),
                target: format!("Expert {} (L{})", expert_id, layer),
                progress: rng.gen_range(0.0..0.2),
            });
        }

        // Occasional log entries
        if self.tick % 8 == 0 {
            let elapsed = self.start_time.elapsed();
            let ts = format!("{:02}:{:02}.{:03}",
                elapsed.as_secs() / 60,
                elapsed.as_secs() % 60,
                elapsed.subsec_millis());

            let messages = [
                (LogLevel::Debug, format!("Router prediction L{}: experts [{}, {}, {}, {}, {}, {}, {}, {}]",
                    rng.gen_range(0..61),
                    rng.gen_range(0..256), rng.gen_range(0..256), rng.gen_range(0..256), rng.gen_range(0..256),
                    rng.gen_range(0..256), rng.gen_range(0..256), rng.gen_range(0..256), rng.gen_range(0..256))),
                (LogLevel::Info, format!("Prefetch hit: expert {} already loaded (temp={:.2})", rng.gen_range(0..256), rng.gen_range(2.0..8.0))),
                (LogLevel::Debug, format!("Temperature decay: {} experts updated", rng.gen_range(10..30))),
                (LogLevel::Warn, format!("Memory pressure {:.1}% — evicting {} cold experts", self.memory_pressure * 100.0, rng.gen_range(1..4))),
                (LogLevel::Info, format!("NVMe stream: expert {} loaded in {:.1}ms ({:.1} GB/s)", rng.gen_range(0..256), rng.gen_range(2.0..15.0), self.nvme_read_gbs)),
            ];

            let (level, message) = &messages[rng.gen_range(0..messages.len())];
            self.logs.push(LogEntry {
                timestamp: ts,
                level: *level,
                message: message.clone(),
            });

            // Keep log bounded
            if self.logs.len() > 100 {
                self.logs.drain(0..self.logs.len() - 80);
            }
        }
    }
}

// ─── UI rendering ────────────────────────────────────────────────────────────

fn ui(frame: &mut Frame, state: &DashboardState) {
    let area = frame.area();

    // Main layout: header + body + footer
    let outer = Layout::vertical([
        Constraint::Length(3),   // header
        Constraint::Min(10),    // body
        Constraint::Length(1),  // footer
    ]).split(area);

    render_header(frame, outer[0], state);
    render_body(frame, outer[1], state);
    render_footer(frame, outer[2], state);
}

fn render_header(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let elapsed = state.start_time.elapsed();
    let uptime = format!("{:02}:{:02}:{:02}",
        elapsed.as_secs() / 3600,
        (elapsed.as_secs() % 3600) / 60,
        elapsed.as_secs() % 60);

    let title = Line::from(vec![
        Span::styled(" ⚡ ExpertFlow ", Style::default().fg(Color::Cyan).bold()),
        Span::styled("│ ", Style::default().fg(Color::DarkGray)),
        Span::styled(&state.config.name, Style::default().fg(Color::White).bold()),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.1} tok/s", state.tokens_per_sec), Style::default().fg(Color::Green).bold()),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("RAM {:.0}/{:.0} GB", state.used_ram_gb, state.total_ram_gb), Style::default().fg(
            if state.memory_pressure > 0.8 { Color::Red }
            else if state.memory_pressure > 0.6 { Color::Yellow }
            else { Color::Green }
        )),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("↑ {}", uptime), Style::default().fg(Color::DarkGray)),
    ]);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .border_type(BorderType::Rounded);

    let para = Paragraph::new(title).block(block).alignment(Alignment::Left);
    frame.render_widget(para, area);
}

fn render_body(frame: &mut Frame, area: Rect, state: &DashboardState) {
    // Top row: System + Expert Cache
    // Bottom row: Scheduler + Token Gen + Model Info
    let rows = Layout::vertical([
        Constraint::Percentage(45),
        Constraint::Percentage(55),
    ]).split(area);

    let top_cols = Layout::horizontal([
        Constraint::Percentage(35),
        Constraint::Percentage(65),
    ]).split(rows[0]);

    let bottom_cols = Layout::horizontal([
        Constraint::Percentage(40),
        Constraint::Percentage(35),
        Constraint::Percentage(25),
    ]).split(rows[1]);

    render_system_panel(frame, top_cols[0], state);
    render_expert_cache_panel(frame, top_cols[1], state);
    render_scheduler_panel(frame, bottom_cols[0], state);
    render_token_panel(frame, bottom_cols[1], state);
    render_model_panel(frame, bottom_cols[2], state);
}

fn panel_border_style(panel_idx: usize, active: usize) -> Style {
    if panel_idx == active {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

fn render_system_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .title(Span::styled(" System ", Style::default().fg(Color::Cyan).bold()))
        .borders(Borders::ALL)
        .border_style(panel_border_style(0, state.active_panel))
        .border_type(BorderType::Rounded);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::vertical([
        Constraint::Length(1), // GPU bar
        Constraint::Length(1), // ANE bar
        Constraint::Length(1), // CPU bar
        Constraint::Length(1), // spacer
        Constraint::Length(1), // Memory
        Constraint::Length(1), // KV cache
        Constraint::Length(1), // Expert budget
        Constraint::Length(1), // spacer
        Constraint::Length(1), // NVMe
        Constraint::Length(1), // Pressure
    ]).split(inner);

    // GPU utilization bar
    let gpu_bar = Gauge::default()
        .label(Span::styled(
            format!("GPU   {:>5.1}%", state.gpu_util * 100.0),
            Style::default().fg(Color::White).bold()
        ))
        .ratio(state.gpu_util)
        .gauge_style(Style::default().fg(Color::Green).bg(Color::DarkGray));
    frame.render_widget(gpu_bar, rows[0]);

    // ANE utilization bar
    let ane_bar = Gauge::default()
        .label(Span::styled(
            format!("ANE   {:>5.1}%", state.ane_util * 100.0),
            Style::default().fg(Color::White).bold()
        ))
        .ratio(state.ane_util)
        .gauge_style(Style::default().fg(Color::Magenta).bg(Color::DarkGray));
    frame.render_widget(ane_bar, rows[1]);

    // CPU utilization bar
    let cpu_bar = Gauge::default()
        .label(Span::styled(
            format!("CPU   {:>5.1}%", state.cpu_util * 100.0),
            Style::default().fg(Color::White).bold()
        ))
        .ratio(state.cpu_util)
        .gauge_style(Style::default().fg(Color::Blue).bg(Color::DarkGray));
    frame.render_widget(cpu_bar, rows[2]);

    // Memory info
    let mem_line = Line::from(vec![
        Span::styled("Memory    ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.1}", state.used_ram_gb), Style::default().fg(Color::White).bold()),
        Span::styled(format!(" / {:.0} GB", state.total_ram_gb), Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(mem_line), rows[4]);

    let kv_line = Line::from(vec![
        Span::styled("KV Cache  ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.1} GB", state.kv_cache_gb), Style::default().fg(Color::Yellow)),
        Span::styled(format!("  ({:.1}%)", state.kv_cache_gb / state.total_ram_gb * 100.0), Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(kv_line), rows[5]);

    let budget_line = Line::from(vec![
        Span::styled("Expert $  ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.1} GB", state.expert_budget_gb), Style::default().fg(Color::Green)),
        Span::styled(format!("  ({:.1}%)", state.expert_budget_gb / state.total_ram_gb * 100.0), Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(budget_line), rows[6]);

    // NVMe bandwidth
    let nvme_line = Line::from(vec![
        Span::styled("NVMe      ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("↓{:.1}", state.nvme_read_gbs), Style::default().fg(Color::Cyan)),
        Span::styled(" / ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("↑{:.1} GB/s", state.nvme_write_gbs), Style::default().fg(Color::DarkGray)),
    ]);
    frame.render_widget(Paragraph::new(nvme_line), rows[8]);

    // Pressure
    let pressure_color = if state.memory_pressure > 0.8 { Color::Red }
        else if state.memory_pressure > 0.6 { Color::Yellow }
        else { Color::Green };
    let pressure_line = Line::from(vec![
        Span::styled("Pressure  ", Style::default().fg(Color::DarkGray)),
        Span::styled(format!("{:.1}%", state.memory_pressure * 100.0), Style::default().fg(pressure_color).bold()),
    ]);
    frame.render_widget(Paragraph::new(pressure_line), rows[9]);
}

fn render_expert_cache_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let total = state.cache_hits + state.cache_misses;
    let hit_rate = if total > 0 { state.cache_hits as f64 / total as f64 * 100.0 } else { 0.0 };

    let title = format!(" Expert Cache — {:.1}% hit rate │ {} evictions ", hit_rate, state.evictions);
    let block = Block::default()
        .title(Span::styled(title, Style::default().fg(Color::Cyan).bold()))
        .borders(Borders::ALL)
        .border_style(panel_border_style(1, state.active_panel))
        .border_type(BorderType::Rounded);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    // Table header + rows
    let header = Row::new(vec![
        Cell::from("ID").style(Style::default().fg(Color::DarkGray).bold()),
        Cell::from("Layer").style(Style::default().fg(Color::DarkGray).bold()),
        Cell::from("State").style(Style::default().fg(Color::DarkGray).bold()),
        Cell::from("Temp").style(Style::default().fg(Color::DarkGray).bold()),
        Cell::from("Size").style(Style::default().fg(Color::DarkGray).bold()),
        Cell::from("Last Access").style(Style::default().fg(Color::DarkGray).bold()),
    ]).height(1);

    let rows: Vec<Row> = state.experts.iter().map(|e| {
        Row::new(vec![
            Cell::from(format!("{:>3}", e.id)).style(Style::default().fg(Color::White)),
            Cell::from(format!("L{:<2}", e.layer)).style(Style::default().fg(Color::DarkGray)),
            Cell::from(format!(" {} ", e.state.label())).style(Style::default().fg(e.state.color()).bold()),
            Cell::from(format!("{:>5.2}", e.temperature)).style(Style::default().fg(e.state.color())),
            Cell::from(format!("{:.0}M", e.size_mb)).style(Style::default().fg(Color::DarkGray)),
            Cell::from(format!("{}ms", e.last_access_ms)).style(Style::default().fg(
                if e.last_access_ms < 100 { Color::Green }
                else if e.last_access_ms < 2000 { Color::Yellow }
                else { Color::DarkGray }
            )),
        ])
    }).collect();

    let widths = [
        Constraint::Length(5),
        Constraint::Length(5),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(7),
        Constraint::Length(10),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    frame.render_widget(table, inner);
}

fn render_scheduler_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .title(Span::styled(" Scheduler ", Style::default().fg(Color::Cyan).bold()))
        .borders(Borders::ALL)
        .border_style(panel_border_style(2, state.active_panel))
        .border_type(BorderType::Rounded);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let layout = Layout::vertical([
        Constraint::Length(state.tasks.len() as u16 + 1), // active tasks
        Constraint::Length(1), // spacer
        Constraint::Min(3),   // log
    ]).split(inner);

    // Active tasks
    let mut task_lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("Active Tasks", Style::default().fg(Color::White).bold()),
        ]),
    ];
    for task in &state.tasks {
        let bar_width = 15;
        let filled = (task.progress * bar_width as f64) as usize;
        let empty = bar_width - filled;
        let bar = format!("{}{}",
            "█".repeat(filled),
            "░".repeat(empty));
        let color = match task.kind.as_str() {
            "GPU" => Color::Green,
            "ANE" => Color::Magenta,
            "CPU" => Color::Blue,
            _ => Color::Cyan,
        };
        task_lines.push(Line::from(vec![
            Span::styled(format!(" {:<8} ", task.kind), Style::default().fg(color).bold()),
            Span::styled(bar.clone(), Style::default().fg(color)),
            Span::styled(format!(" {}", task.target), Style::default().fg(Color::DarkGray)),
        ]));
    }
    frame.render_widget(Paragraph::new(task_lines), layout[0]);

    // Log
    let visible_logs = if state.logs.len() > layout[2].height as usize {
        &state.logs[state.logs.len() - layout[2].height as usize..]
    } else {
        &state.logs
    };

    let log_lines: Vec<Line> = visible_logs.iter().map(|entry| {
        Line::from(vec![
            Span::styled(format!("{} ", entry.timestamp), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{} ", entry.level.label()), Style::default().fg(entry.level.color()).bold()),
            Span::styled(&entry.message, Style::default().fg(Color::White)),
        ])
    }).collect();

    let log_block = Block::default()
        .title(Span::styled(" Log ", Style::default().fg(Color::DarkGray)))
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));

    frame.render_widget(Paragraph::new(log_lines).block(log_block), layout[2]);
}

fn render_token_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .title(Span::styled(" Token Generation ", Style::default().fg(Color::Cyan).bold()))
        .borders(Borders::ALL)
        .border_style(panel_border_style(3, state.active_panel))
        .border_type(BorderType::Rounded);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let layout = Layout::vertical([
        Constraint::Length(4), // stats
        Constraint::Length(1), // spacer
        Constraint::Min(3),   // current text
    ]).split(inner);

    // Stats
    let tps_color = if state.tokens_per_sec > 20.0 { Color::Green }
        else if state.tokens_per_sec > 10.0 { Color::Yellow }
        else { Color::Red };

    let stats = vec![
        Line::from(vec![
            Span::styled("Throughput  ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1} tok/s", state.tokens_per_sec), Style::default().fg(tps_color).bold()),
        ]),
        Line::from(vec![
            Span::styled("Latency     ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1} ms/tok", state.latency_ms), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Generated   ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", state.tokens_generated), Style::default().fg(Color::White).bold()),
            Span::styled(format!("  (prompt: {})", state.prompt_tokens), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Cache Hit   ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", state.cache_hits), Style::default().fg(Color::Green)),
            Span::styled(" / ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", state.cache_misses), Style::default().fg(Color::Red)),
            Span::styled(" miss", Style::default().fg(Color::DarkGray)),
        ]),
    ];
    frame.render_widget(Paragraph::new(stats), layout[0]);

    // Current text with blinking cursor effect
    let cursor = if state.tick % 4 < 2 { "█" } else { " " };
    let text = format!("{}{}", state.current_text, cursor);
    let text_block = Block::default()
        .title(Span::styled(" Output ", Style::default().fg(Color::DarkGray)))
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));

    frame.render_widget(
        Paragraph::new(text)
            .wrap(Wrap { trim: true })
            .style(Style::default().fg(Color::White))
            .block(text_block),
        layout[2],
    );
}

fn render_model_panel(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let block = Block::default()
        .title(Span::styled(" Model ", Style::default().fg(Color::Cyan).bold()))
        .borders(Borders::ALL)
        .border_style(panel_border_style(4, state.active_panel))
        .border_type(BorderType::Rounded);

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let config = &state.config;
    let total_gb = config.total_size() as f64 / 1e9;
    let active_gb = config.active_size_per_layer() as f64 / 1e9;
    let expert_mb = config.expert_size_bytes as f64 / 1e6;

    let quant = if config.name.contains("Q4") { "Q4_K_M" }
        else if config.name.contains("Q2") { "Q2_K" }
        else if config.name.contains("1bit") { "1-bit" }
        else { "FP16" };

    let lines = vec![
        Line::from(vec![
            Span::styled(&config.name, Style::default().fg(Color::White).bold()),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Layers      ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", config.num_layers), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Experts     ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", config.num_experts), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Active/tok  ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", config.num_active_experts), Style::default().fg(Color::Green).bold()),
        ]),
        Line::from(vec![
            Span::styled("Quant       ", Style::default().fg(Color::DarkGray)),
            Span::styled(quant, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Expert size ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.0} MB", expert_mb), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Total       ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.0} GB", total_gb), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Active/layer", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.1} GB", active_gb), Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Hidden      ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", config.hidden_size), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("FFN         ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", config.intermediate_size), Style::default().fg(Color::DarkGray)),
        ]),
    ];

    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_footer(frame: &mut Frame, area: Rect, state: &DashboardState) {
    let panels = ["System", "Cache", "Scheduler", "Tokens", "Model"];
    let mut spans = vec![Span::styled(" ", Style::default())];

    for (i, name) in panels.iter().enumerate() {
        if i == state.active_panel {
            spans.push(Span::styled(format!("[{}]", name), Style::default().fg(Color::Cyan).bold()));
        } else {
            spans.push(Span::styled(format!(" {} ", name), Style::default().fg(Color::DarkGray)));
        }
    }

    spans.push(Span::styled("  │  ", Style::default().fg(Color::DarkGray)));
    spans.push(Span::styled("Tab", Style::default().fg(Color::Yellow).bold()));
    spans.push(Span::styled(" switch  ", Style::default().fg(Color::DarkGray)));
    spans.push(Span::styled("q", Style::default().fg(Color::Yellow).bold()));
    spans.push(Span::styled(" quit", Style::default().fg(Color::DarkGray)));

    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = DashboardState::new();
    let tick_rate = Duration::from_millis(150);

    loop {
        terminal.draw(|frame| ui(frame, &state))?;

        if event::poll(tick_rate)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Tab => {
                            state.active_panel = (state.active_panel + 1) % state.panel_count;
                        }
                        KeyCode::BackTab => {
                            state.active_panel = if state.active_panel == 0 {
                                state.panel_count - 1
                            } else {
                                state.active_panel - 1
                            };
                        }
                        KeyCode::Char('1') => state.active_panel = 0,
                        KeyCode::Char('2') => state.active_panel = 1,
                        KeyCode::Char('3') => state.active_panel = 2,
                        KeyCode::Char('4') => state.active_panel = 3,
                        KeyCode::Char('5') => state.active_panel = 4,
                        _ => {}
                    }
                }
            }
        }

        state.tick_update();
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
