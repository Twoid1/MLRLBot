"""
Real-Time Training Progress Monitor
- Live updates during training
- Visual progress bars
- Performance metrics
- ETA estimation
- Web dashboard option
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
import sys

# Rich terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.layout import Layout
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install 'rich' for better visuals: pip install rich")


class TrainingMonitor:
    """
    Real-time monitoring of training progress
    Can be run in separate terminal while training executes
    """
    
    def __init__(self, progress_file: str = 'logs/training_progress.json'):
        self.progress_file = Path(progress_file)
        self.console = Console() if RICH_AVAILABLE else None
        self.last_update = None
        
    def monitor_live(self, refresh_rate: float = 2.0):
        """
        Live monitoring with auto-refresh
        Run this in a separate terminal while training
        """
        if not RICH_AVAILABLE:
            # Fallback to simple text monitoring
            self._monitor_simple(refresh_rate)
            return
        
        with Live(self._generate_display(), refresh_per_second=1/refresh_rate, console=self.console) as live:
            while True:
                try:
                    display = self._generate_display()
                    live.update(display)
                    time.sleep(refresh_rate)
                    
                    # Check if training is complete
                    if self._is_training_complete():
                        self.console.print("\n[bold green] Training Complete![/bold green]")
                        break
                        
                except KeyboardInterrupt:
                    self.console.print("\n[yellow]Monitoring stopped[/yellow]")
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {e}[/red]")
                    time.sleep(refresh_rate)
    
    def _generate_display(self):
        """Generate rich display layout"""
        if not self.progress_file.exists():
            return Panel("[yellow]Waiting for training to start...[/yellow]", title="Training Monitor")
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
        except:
            return Panel("[red]Error reading progress file[/red]", title="Training Monitor")
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Header
        start_time = datetime.fromisoformat(data.get('start_time', datetime.now().isoformat()))
        elapsed = datetime.now() - start_time
        
        header_text = f"[bold cyan]Training Progress Monitor[/bold cyan]\n"
        header_text += f"Elapsed: {elapsed.seconds // 3600}h {(elapsed.seconds % 3600) // 60}m {elapsed.seconds % 60}s"
        layout["header"].update(Panel(header_text, style="cyan"))
        
        # Main content - stages table
        table = Table(title="Training Stages", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Stage", style="cyan", width=20)
        table.add_column("Status", width=12)
        table.add_column("Progress", width=30)
        table.add_column("Time", width=10)
        table.add_column("Metrics", width=30)
        
        stages_info = data.get('stages', {})
        stage_order = ['Data Loading', 'Feature Engineering', 'Data Splitting', 
                      'ML Training', 'RL Training', 'Saving Models']
        
        for stage_name in stage_order:
            if stage_name not in stages_info:
                table.add_row(stage_name, "[dim]pending[/dim]", "", "", "")
                continue
            
            stage = stages_info[stage_name]
            status = stage.get('status', 'unknown')
            
            # Status emoji
            if status == 'completed':
                status_icon = "[green] complete[/green]"
            elif status == 'running':
                status_icon = "[yellow] running[/yellow]"
            else:
                status_icon = "[dim]pending[/dim]"
            
            # Progress bar
            progress = stage.get('progress', 0)
            completed = stage.get('completed_items', 0)
            total = stage.get('total_items', 100)
            
            if status == 'completed':
                bar = "[green]" + "l" * 20 + "[/green]"
                progress_text = f"{bar} 100%"
            elif status == 'running':
                filled = int(progress / 5)
                bar = "[yellow]" + "l" * filled + "[/yellow]" + "i" * (20 - filled)
                progress_text = f"{bar} {progress:.1f}% ({completed}/{total})"
            else:
                progress_text = ""
            
            # Time
            if 'total_time' in stage:
                time_str = f"{stage['total_time']:.1f}s"
            elif 'elapsed_time' in stage:
                time_str = f"{stage['elapsed_time']:.1f}s"
            else:
                time_str = ""
            
            # Metrics
            metrics_str = ""
            if 'metrics' in stage:
                metrics = stage['metrics']
                if 'current_reward' in metrics:
                    metrics_str = f"Reward: {metrics['current_reward']:.2f}"
                if 'accuracy' in metrics:
                    metrics_str = f"Acc: {metrics['accuracy']:.4f}"
            elif 'final_metrics' in stage:
                fm = stage['final_metrics']
                if 'accuracy' in fm:
                    metrics_str = f"Acc: {fm['accuracy']:.4f}"
                if 'datasets_loaded' in fm:
                    metrics_str = f"Datasets: {fm['datasets_loaded']}"
            
            table.add_row(stage_name, status_icon, progress_text, time_str, metrics_str)
        
        layout["main"].update(Panel(table, title="Progress Details", border_style="blue"))
        
        # Footer - current stage info
        current_stage = data.get('current_stage', 'None')
        footer_text = f"[bold]Current Stage:[/bold] [yellow]{current_stage}[/yellow]"
        
        if current_stage and current_stage in stages_info:
            stage = stages_info[current_stage]
            if 'metrics' in stage:
                metrics = stage['metrics']
                footer_text += "\n"
                if 'current_reward' in metrics:
                    footer_text += f"  Reward: {metrics['current_reward']:.2f}"
                    if 'avg_reward_last_10' in metrics:
                        footer_text += f" | Avg(10): {metrics['avg_reward_last_10']:.2f}"
                    if 'epsilon' in metrics:
                        footer_text += f" | E: {metrics['epsilon']:.3f}"
        
        layout["footer"].update(Panel(footer_text, style="green"))
        
        return layout
    
    def _monitor_simple(self, refresh_rate: float = 2.0):
        """Simple text-based monitoring (fallback)"""
        print("="*60)
        print("Training Progress Monitor (Simple Mode)")
        print("="*60)
        print("Install 'rich' for better visuals: pip install rich")
        print("="*60)
        
        while True:
            try:
                if not self.progress_file.exists():
                    print("\rWaiting for training to start...", end="")
                    time.sleep(refresh_rate)
                    continue
                
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                
                # Clear screen
                print("\033[2J\033[H")  # ANSI escape codes
                print("="*60)
                print("Training Progress Monitor")
                print("="*60)
                
                start_time = datetime.fromisoformat(data.get('start_time', datetime.now().isoformat()))
                elapsed = datetime.now() - start_time
                print(f"Elapsed: {elapsed.seconds // 3600}h {(elapsed.seconds % 3600) // 60}m {elapsed.seconds % 60}s")
                print("-"*60)
                
                stages = data.get('stages', {})
                for stage_name, stage_info in stages.items():
                    status = stage_info.get('status', 'unknown')
                    progress = stage_info.get('progress', 0)
                    
                    status_icon = "Y" if status == "completed" else "..." if status == "running" else "o"
                    print(f"{status_icon} {stage_name:.<30} {progress:5.1f}%")
                    
                    if 'metrics' in stage_info:
                        for key, value in stage_info['metrics'].items():
                            print(f"    {key}: {value}")
                
                print("="*60)
                print(f"Current: {data.get('current_stage', 'None')}")
                
                if self._is_training_complete():
                    print("\n Training Complete!")
                    break
                
                time.sleep(refresh_rate)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(refresh_rate)
    
    def _is_training_complete(self) -> bool:
        """Check if training is complete"""
        if not self.progress_file.exists():
            return False
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            
            stages = data.get('stages', {})
            if not stages:
                return False
            
            # Check if all stages are completed
            return all(s.get('status') == 'completed' for s in stages.values())
            
        except:
            return False
    
    def generate_summary_report(self, output_file: Optional[str] = None) -> str:
        """Generate a final summary report"""
        if not self.progress_file.exists():
            return "No training data available"
        
        with open(self.progress_file, 'r') as f:
            data = json.load(f)
        
        report = []
        report.append("="*80)
        report.append("TRAINING SUMMARY REPORT")
        report.append("="*80)
        report.append("")
        
        # Overall timing
        start_time = datetime.fromisoformat(data['start_time'])
        stages = data.get('stages', {})
        
        total_time = sum(s.get('total_time', 0) for s in stages.values() if 'total_time' in s)
        report.append(f"Total Training Time: {total_time/3600:.2f} hours")
        report.append(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Stage breakdown
        report.append("Stage Breakdown:")
        report.append("-" * 80)
        
        for stage_name, stage_info in stages.items():
            status = stage_info.get('status', 'unknown')
            time_taken = stage_info.get('total_time', stage_info.get('elapsed_time', 0))
            
            report.append(f"\n{stage_name}:")
            report.append(f"  Status: {status}")
            report.append(f"  Time: {time_taken:.2f}s ({time_taken/60:.2f} min)")
            
            if 'final_metrics' in stage_info:
                report.append("  Final Metrics:")
                for key, value in stage_info['final_metrics'].items():
                    report.append(f"    {key}: {value}")
        
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text


def monitor_training(progress_file: str = 'logs/training_progress.json', refresh_rate: float = 2.0):
    """
    Main function to monitor training
    
    Usage:
        # In separate terminal while training runs:
        python training_monitor.py
    """
    monitor = TrainingMonitor(progress_file)
    monitor.monitor_live(refresh_rate)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor training progress in real-time')
    parser.add_argument('--file', default='logs/training_progress.json', help='Progress file to monitor')
    parser.add_argument('--refresh', type=float, default=2.0, help='Refresh rate in seconds')
    parser.add_argument('--report', action='store_true', help='Generate summary report')
    parser.add_argument('--output', help='Output file for report')
    
    args = parser.parse_args()
    
    if args.report:
        monitor = TrainingMonitor(args.file)
        report = monitor.generate_summary_report(args.output)
        print(report)
    else:
        monitor_training(args.file, args.refresh)