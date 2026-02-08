#!/usr/bin/env python3
"""Check Run 86 stats from database."""

from src.data.database import get_db_session, TrainingRun, TrainingMetric

with get_db_session() as session:
    run = session.query(TrainingRun).filter_by(run_id=86).first()
    
    if not run:
        print("Run 86 not found")
    else:
        print("Run 86 Stats:")
        print(f"  Win Rate: {run.final_win_rate if run.final_win_rate else 'N/A'}%")
        print(f"  Mean Reward: {run.mean_reward if run.mean_reward else 'N/A'}")
        print(f"  Sharpe: {run.sharpe_ratio if run.sharpe_ratio else 'N/A'}")
        print(f"  Status: {run.status}")
        
        metrics = session.query(TrainingMetric).filter_by(run_id=86).order_by(TrainingMetric.step).all()
        
        if metrics:
            print(f"\nLast 5 checkpoints:")
            for m in metrics[-5:]:
                buy = m.count_buy if hasattr(m, 'count_buy') and m.count_buy else 0
                sell = m.count_sell if hasattr(m, 'count_sell') and m.count_sell else 0  
                no_act = m.count_no_action if hasattr(m, 'count_no_action') and m.count_no_action else 0
                print(f"  Step {m.step}: BUY={buy}, SELL={sell}, NO_ACTION={no_act}, Win={m.win_rate if m.win_rate else 0:.1f}%")
