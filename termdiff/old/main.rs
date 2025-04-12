use clap::{App as ClapApp, Arg};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Span, Line},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::{
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader},
};

struct FileContent {
    path: String,
    lines: Vec<String>,
}

struct App {
    files: Vec<FileContent>,
    scroll: usize,
    highlight_diffs: bool,
}

impl App {
    fn new(file_paths: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let mut files = Vec::new();

        for path in file_paths {
            let content = read_file_lines(&path)?;
            files.push(FileContent {
                path,
                lines: content,
            });
        }

        Ok(App {
            files,
            scroll: 0,
            highlight_diffs: true,
        })
    }

    fn scroll_up(&mut self) {
        if self.scroll > 0 {
            self.scroll -= 1;
        }
    }

    fn scroll_down(&mut self) {
        let max_lines = self.files.iter().map(|f| f.lines.len()).max().unwrap_or(0);
        if self.scroll < max_lines.saturating_sub(1) {
            self.scroll += 1;
        }
    }

    fn toggle_highlight(&mut self) {
        self.highlight_diffs = !self.highlight_diffs;
    }
}

fn read_file_lines(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines: Result<Vec<String>, _> = reader.lines().collect();
    Ok(lines?)
}

fn are_lines_different(lines: &[&str]) -> bool {
    if lines.len() <= 1 {
        return false;
    }

    let first = lines[0];
    for line in lines.iter().skip(1) {
        if *line != first {
            return true;
        }
    }
    false
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
) -> Result<(), Box<dyn Error>> {
    loop {
        terminal.draw(|f| {
            let size = f.size();

            // Create horizontal layout for file panels
            let constraints: Vec<Constraint> =
                vec![Constraint::Percentage(100 / app.files.len() as u16); app.files.len()];

            let horizontal_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(constraints)
                .split(size);

            for (idx, file) in app.files.iter().enumerate() {
                let block = Block::default()
                    .title(format!(" {} ", file.path))
                    .borders(Borders::ALL);

                let height = horizontal_chunks[idx].height.saturating_sub(2) as usize; // Account for borders
                let file_lines = file.lines.len();

                let _display_range = app.scroll..std::cmp::min(app.scroll + height, file_lines);
                let visible_lines: Vec<&str> = file
                    .lines
                    .iter()
                    .skip(app.scroll)
                    .take(height)
                    .map(|s| s.as_str())
                    .collect();

                let spans_vec: Vec<Line> = visible_lines
                    .iter()
                    .enumerate()
                    .map(|(i, &line)| {
                        let line_idx = app.scroll + i;

                        // Check if this line exists in all files and if they're different
                        let mut all_lines = Vec::new();
                        for f in &app.files {
                            if line_idx < f.lines.len() {
                                all_lines.push(f.lines[line_idx].as_str());
                            } else {
                                all_lines.push("");
                            }
                        }

                        let is_diff = app.highlight_diffs && are_lines_different(&all_lines);

                        // Line number and content
                        let line_num = format!("{:4} | ", line_idx + 1);

                        if is_diff {
                            let line_num_span =
                                Span::styled(line_num, Style::default().fg(Color::Gray));

                            let content_span = Span::styled(
                                line,
                                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                            );

                            Line::from(vec![line_num_span, content_span])
                        } else {
                            let line_num_span =
                                Span::styled(line_num, Style::default().fg(Color::Gray));

                            let content_span = Span::styled(line, Style::default());

                            Line::from(vec![line_num_span, content_span])
                        }
                    })
                    .collect();

                let paragraph = Paragraph::new(spans_vec).block(block);

                f.render_widget(paragraph, horizontal_chunks[idx]);
            }
        })?;

        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') => return Ok(()),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Ok(())
                }
                KeyCode::Up | KeyCode::Char('k') => app.scroll_up(),
                KeyCode::Down | KeyCode::Char('j') => app.scroll_down(),
                KeyCode::PageUp => {
                    for _ in 0..10 {
                        app.scroll_up();
                    }
                }
                KeyCode::PageDown => {
                    for _ in 0..10 {
                        app.scroll_down();
                    }
                }
                KeyCode::Home => app.scroll = 0,
                KeyCode::End => {
                    let max_lines = app.files.iter().map(|f| f.lines.len()).max().unwrap_or(0);
                    app.scroll = max_lines.saturating_sub(1);
                }
                KeyCode::Char('h') => app.toggle_highlight(),
                _ => {}
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let matches = ClapApp::new("File Diff Viewer")
        .version("1.0")
        .author("Rust Developer")
        .about("Compare 2 or 3 files side by side")
        .arg(
            Arg::with_name("files")
                .help("Files to compare")
                .required(true)
                .min_values(2)
                .max_values(3),
        )
        .get_matches();

    let file_paths: Vec<String> = matches
        .values_of("files")
        .unwrap()
        .map(|s| s.to_string())
        .collect();

    if file_paths.len() < 2 || file_paths.len() > 3 {
        eprintln!("Please provide 2 or 3 files to compare");
        return Ok(());
    }

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app and run it
    let app = App::new(file_paths)?;
    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        eprintln!("{:?}", err);
    }

    Ok(())
}
