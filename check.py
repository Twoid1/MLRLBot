"""
Debug script to check if trades have duration data
"""
import pickle
import glob

# Find the most recent episode results
result_files = glob.glob('results/episode_results_*.pkl')
if not result_files:
    print("No episode results found in results/ directory")
    exit(1)

latest_file = max(result_files, key=lambda x: x.split('_')[-1])
print(f"Loading: {latest_file}\n")

with open(latest_file, 'rb') as f:
    episode_results = pickle.load(f)

print(f"Total episodes: {len(episode_results)}\n")

# Check first episode with trades
for i, episode in enumerate(episode_results[:10]):  # Check first 10 episodes
    if 'trades' in episode and episode['trades']:
        print(f"Episode {episode.get('episode', i+1)}:")
        print(f"  Asset: {episode.get('asset', 'N/A')}")
        print(f"  Timeframe: {episode.get('timeframe', 'N/A')}")
        print(f"  Total trades: {len(episode['trades'])}\n")
        
        # Check each trade
        for j, trade in enumerate(episode['trades'][:5]):  # Show first 5 trades
            print(f"  Trade {j+1}:")
            print(f"    Action: {trade.get('action', 'N/A')}")
            print(f"    PnL: {trade.get('pnl', 'N/A')}")
            print(f"    Duration: {trade.get('duration', 'N/A')}")
            print(f"    Has duration: {'YES' if trade.get('duration') is not None else 'NO'}")
            print()
        
        if len(episode['trades']) > 5:
            print(f"  ... and {len(episode['trades']) - 5} more trades\n")
        
        break
else:
    print("No episodes with trades found in first 10 episodes!")

# Count trades with duration across all episodes
total_trades = 0
trades_with_duration = 0
trades_with_pnl = 0
closing_trades = 0

for episode in episode_results:
    if 'trades' in episode:
        for trade in episode['trades']:
            total_trades += 1
            if trade.get('duration') is not None:
                trades_with_duration += 1
            if trade.get('pnl') is not None and trade.get('pnl') != 0:
                trades_with_pnl += 1
            if trade.get('action') in ['SELL', 'BUY_COVER', 'COVER']:
                closing_trades += 1

print("=" * 60)
print("SUMMARY STATISTICS:")
print("=" * 60)
print(f"Total trades across all episodes: {total_trades}")
print(f"Trades with PnL: {trades_with_pnl}")
print(f"Trades with duration: {trades_with_duration}")
print(f"Closing actions (SELL/BUY_COVER/COVER): {closing_trades}")
print(f"\nDuration coverage: {trades_with_duration/total_trades*100:.1f}%" if total_trades > 0 else "\nNo trades found!")