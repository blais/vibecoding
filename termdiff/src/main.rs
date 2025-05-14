use clap::{Command as ClapCommand, Arg};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::{
    // collections::HashMap,
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader},
    process::Command,
};

// xxdiff color scheme
struct XxdiffColors {
    same: Color,        // Same content across files
    different: Color,   // Different content
    insert: Color,      // Line only in one file
    delete: Color,      // Line missing in a file
    text_normal: Color, // Normal text color
    background: Color,  // Background color for all text
}

impl Default for XxdiffColors {
    fn default() -> Self {
        XxdiffColors {
            same: Color::Black,                   // Default text color
            different: Color::Rgb(95, 0, 0,), // Light red, similar to xxdiff
            insert: Color::Rgb(0, 95, 0),    // Light green
            delete: Color::Rgb(0, 0, 95),    // Light lavender
            text_normal: Color::Gray, // inverted
            background: Color::Rgb(100,100,100), // Light gray background
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
#[allow(dead_code)]
enum LineStatus {
    Same,
    Different,
    OnlyInFile(usize), // Only in file with index
    Missing,           // Line is missing in this file
}

struct FileContent {
    path: String,
    lines: Vec<String>,
    line_status: Vec<LineStatus>,
}

struct App {
    files: Vec<FileContent>,
    scroll: usize,
    highlight_diffs: bool,
    colors: XxdiffColors,
}

impl App {
    fn new(file_paths: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let mut files = Vec::new();

        for path in file_paths.iter() {
            let content = read_file_lines(path)?;
            files.push(FileContent {
                path: path.clone(),
                lines: content,
                line_status: Vec::new(), // Will be populated after diff
            });
        }

        let mut app = App {
            files,
            scroll: 0,
            highlight_diffs: true,
            colors: XxdiffColors::default(),
        };

        // Compute diffs using external diff tool
        app.compute_diffs()?;

        Ok(app)
    }

    fn compute_diffs(&mut self) -> Result<(), Box<dyn Error>> {
        if self.files.len() < 2 {
            return Ok(());
        }

        // For each pair of files, run diff and collect results
        for i in 0..self.files.len() {
            // Initialize all lines as "same" first
            self.files[i].line_status = vec![LineStatus::Same; self.files[i].lines.len()];
        }

        // Compare each file with the first file
        let base_file = self.files[0].path.clone();

        for i in 1..self.files.len() {
            let comparison_file = self.files[i].path.clone();

            // Run the diff command
            let output = Command::new("diff")
                .arg("-u")
                .arg(&base_file)
                .arg(comparison_file)
                .output()?;

            if !output.status.success() {
                // Parse the diff output
                let diff_output = String::from_utf8_lossy(&output.stdout);
                self.parse_diff_output(&diff_output, 0, i)?;
            }
        }

        Ok(())
    }

    fn parse_diff_output(
        &mut self,
        diff_output: &str,
        file1_idx: usize,
        file2_idx: usize,
    ) -> Result<(), Box<dyn Error>> {
        let lines: Vec<&str> = diff_output.lines().collect();

        // Skip the first 2 lines (diff header)
        let mut i = 2;

        while i < lines.len() {
            let line = lines[i];

            // Check for hunk headers like @@ -1,7 +1,6 @@
            if line.starts_with("@@") {
                // Extract line numbers
                let parts: Vec<&str> = line.split("@@").collect();
                if parts.len() >= 2 {
                    let range_info = parts[1].trim();
                    let ranges: Vec<&str> = range_info.split(' ').collect();

                    if ranges.len() >= 2 {
                        let file1_range = parse_range(ranges[0])?;
                        let file2_range = parse_range(ranges[1])?;

                        // Process the hunk
                        i = self.process_hunk(
                            &lines,
                            i + 1,
                            file1_idx,
                            file2_idx,
                            file1_range,
                            file2_range,
                        )?;
                        continue;
                    }
                }
            }

            i += 1;
        }

        Ok(())
    }

    fn process_hunk(
        &mut self,
        lines: &Vec<&str>,
        start_idx: usize,
        file1_idx: usize,
        file2_idx: usize,
        file1_range: (usize, usize),
        file2_range: (usize, usize),
    ) -> Result<usize, Box<dyn Error>> {
        let mut file1_line = file1_range.0;
        let mut file2_line = file2_range.0;
        let mut i = start_idx;

        while i < lines.len() {
            let line = lines[i];

            // If we hit another hunk header, return
            if line.starts_with("@@") {
                return Ok(i);
            }

            if line.starts_with('-') {
                // Line only in file1
                if file1_line <= self.files[file1_idx].lines.len() {
                    self.files[file1_idx].line_status[file1_line - 1] = LineStatus::Different;
                }

                // Insert blank line in file2 if needed for alignment
                if file2_line <= self.files[file2_idx].lines.len() && self.files[file2_idx].line_status[file2_line-1] == LineStatus::Same {
                    self.files[file2_idx].lines.insert(file2_line - 1, "".to_string());
                    self.files[file2_idx].line_status.insert(file2_line - 1, LineStatus::Missing);
                }
                file1_line += 1;


            } else if line.starts_with('+') {
                // Line only in file2
                if file2_line <= self.files[file2_idx].lines.len() {
                    self.files[file2_idx].line_status[file2_line - 1] = LineStatus::Different;
                }

                // Insert blank line in file1 if needed for alignment
                if file1_line <= self.files[file1_idx].lines.len() && self.files[file1_idx].line_status[file1_line-1] == LineStatus::Same {

                    self.files[file1_idx].lines.insert(file1_line - 1, "".to_string());
                    self.files[file1_idx].line_status.insert(file1_line - 1, LineStatus::Missing);
                }
                file2_line += 1;

            } else {
                // Line in both files
                if file1_line <= self.files[file1_idx].lines.len() {
                    // Only mark as same if it wasn't already marked as different
                    if self.files[file1_idx].line_status[file1_line - 1] != LineStatus::Different {
                        self.files[file1_idx].line_status[file1_line - 1] = LineStatus::Same;
                    }
                }
                if file2_line <= self.files[file2_idx].lines.len() {
                    if self.files[file2_idx].line_status[file2_line - 1] != LineStatus::Different {
                        self.files[file2_idx].line_status[file2_line - 1] = LineStatus::Same;
                    }
                }
                file1_line += 1;
                file2_line += 1;
            }

            i += 1;
        }

        Ok(i)
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

fn parse_range(range_str: &str) -> Result<(usize, usize), Box<dyn Error>> {
    // Parse range like "-1,7" to (line_start, line_count)
    let clean_range = range_str.trim_start_matches('-').trim_start_matches('+');
    let parts: Vec<&str> = clean_range.split(',').collect();

    if parts.len() >= 2 {
        let start = parts[0].parse::<usize>()?;
        let count = parts[1].parse::<usize>()?;
        Ok((start, count))
    } else if parts.len() == 1 {
        let start = parts[0].parse::<usize>()?;
        Ok((start, 1)) // Assume count of 1 if not specified
    } else {
        Ok((0, 0)) // Default
    }
}

fn read_file_lines(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines: Result<Vec<String>, _> = reader.lines().collect();
    Ok(lines?)
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
) -> Result<(), Box<dyn Error>> {
    loop {
        terminal.draw(|f| {
            let size = f.area();

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
                let visible_lines: Vec<(&str, LineStatus)> = file
                    .lines
                    .iter()
                    .zip(file.line_status.iter())
                    .skip(app.scroll)
                    .take(height)
                    .map(|(s, status)| (s.as_str(), *status))
                    .collect();

                let spans_vec: Vec<Line> = visible_lines
                    .iter()
                    .enumerate()
                    .map(|(i, &(line, status))| {
                        let line_idx = app.scroll + i;

                        // Line number and content
                        let line_num = format!("{:4} | ", line_idx + 1);

                        // Line number with light gray background
                        let line_num_span = Span::styled(
                            line_num,
                            Style::default()
                                .fg(app.colors.text_normal)
                                .bg(app.colors.background),
                        );

                        // Choose color based on line status and if highlighting is enabled
                        let content_style = if app.highlight_diffs {
                            match status {
                                LineStatus::Same => Style::default()
                                    .fg(app.colors.same)
                                    .bg(app.colors.background),
                                LineStatus::Different => Style::default()
                                    .fg(app.colors.different)
                                    .bg(app.colors.background)
                                    .add_modifier(Modifier::BOLD),
                                LineStatus::OnlyInFile(_) => Style::default()
                                    .fg(app.colors.insert)
                                    .bg(app.colors.background)
                                    .add_modifier(Modifier::BOLD),
                                LineStatus::Missing => Style::default()
                                    .fg(app.colors.delete)
                                    .bg(app.colors.background)
                                    .add_modifier(Modifier::ITALIC),
                            }
                        } else {
                            Style::default().bg(app.colors.background)
                        };

                        let content_span = Span::styled(line, content_style);
                        Line::from(vec![line_num_span, content_span])
                    })
                    .collect();

                // Apply style with background to the entire paragraph's block
                let block = block.style(Style::default().bg(app.colors.background));
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
                KeyCode::Char('r') => {
                    // Recompute diffs
                    match app.compute_diffs() {
                        Ok(_) => {}
                        Err(e) => eprintln!("Error recomputing diffs: {:?}", e),
                    }
                }
                _ => {}
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let matches = ClapCommand::new("File Diff Viewer")
        .version("1.0")
        .author("Rust Developer")
        .about("Compare 2 or 3 files side by side")
        .arg(
            Arg::new("files")
                .help("Files to compare")
                .required(true)
                .num_args(2..=3)
        )
        .get_matches();

    let file_paths: Vec<String> = matches
        .get_many::<String>("files")
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
