"""
Order Reconciliation System
Ensures portfolio state matches Kraken actual state
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)


@dataclass
class UnreconciledOrder:
    """Record of order that filled on exchange but not tracked locally"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    fill_price: float
    fill_size: float
    timestamp: datetime
    reconciled: bool = False


class OrderReconciliation:
    """
    Handles order reconciliation between Kraken and local portfolio
    
    Features:
    - Tracks unreconciled orders
    - Automatic retry logic
    - State persistence
    - Recovery mechanisms
    """
    
    def __init__(self, state_file: str = 'logs/unreconciled_orders.json'):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing unreconciled orders
        self.unreconciled_orders: List[UnreconciledOrder] = []
        self._load_state()
    
    def _load_state(self):
        """Load unreconciled orders from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                
                for order_data in data:
                    # Convert timestamp string back to datetime
                    order_data['timestamp'] = datetime.fromisoformat(order_data['timestamp'])
                    self.unreconciled_orders.append(UnreconciledOrder(**order_data))
                
                logger.info(f"Loaded {len(self.unreconciled_orders)} unreconciled orders")
                
            except Exception as e:
                logger.error(f"Error loading reconciliation state: {e}")
    
    def _save_state(self):
        """Save unreconciled orders to disk"""
        try:
            data = []
            for order in self.unreconciled_orders:
                order_dict = asdict(order)
                # Convert datetime to string for JSON
                order_dict['timestamp'] = order.timestamp.isoformat()
                data.append(order_dict)
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving reconciliation state: {e}")
    
    def record_unreconciled_order(self, order_id: str, symbol: str, side: str,
                                   fill_price: float, fill_size: float):
        """Record order that filled on exchange but failed to update portfolio"""
        
        logger.critical("="*80)
        logger.critical("  UNRECONCILED ORDER DETECTED")
        logger.critical("="*80)
        logger.critical(f"Order ID: {order_id}")
        logger.critical(f"Symbol: {symbol}")
        logger.critical(f"Side: {side.upper()}")
        logger.critical(f"Fill Price: ${fill_price:.2f}")
        logger.critical(f"Fill Size: {fill_size:.8f}")
        logger.critical(f"Value: ${fill_price * fill_size:.2f}")
        logger.critical("="*80)
        logger.critical("This order filled on Kraken but is NOT tracked locally!")
        logger.critical("Automatic reconciliation will be attempted.")
        logger.critical("="*80)
        
        unreconciled = UnreconciledOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            fill_price=fill_price,
            fill_size=fill_size,
            timestamp=datetime.now(),
            reconciled=False
        )
        
        self.unreconciled_orders.append(unreconciled)
        self._save_state()
    
    def attempt_reconciliation(self, portfolio, max_retries: int = 3) -> bool:
        """
        Attempt to reconcile all unreconciled orders
        
        Returns:
            True if all orders reconciled, False otherwise
        """
        if not self.unreconciled_orders:
            return True
        
        logger.info(f"Attempting to reconcile {len(self.unreconciled_orders)} order(s)...")
        
        all_reconciled = True
        
        for order in self.unreconciled_orders:
            if order.reconciled:
                continue
            
            logger.info(f"Reconciling order {order.order_id}...")
            
            success = False
            for attempt in range(max_retries):
                try:
                    if order.side == 'buy':
                        success = portfolio.open_position(
                            symbol=order.symbol,
                            quantity=order.fill_size,
                            price=order.fill_price,
                            position_type='long',
                            fees=order.fill_size * order.fill_price * 0.0026
                        )
                    else:  # sell
                        success = portfolio.close_position(
                            symbol=order.symbol,
                            price=order.fill_price,
                            fees=order.fill_size * order.fill_price * 0.0026,
                            reason='reconciliation'
                        )
                    
                    if success:
                        logger.info(f" Successfully reconciled {order.order_id}")
                        order.reconciled = True
                        break
                    else:
                        logger.warning(f"Reconciliation attempt {attempt+1} failed for {order.order_id}")
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Reconciliation error (attempt {attempt+1}): {e}")
                    time.sleep(1)
            
            if not success:
                logger.error(f" Failed to reconcile {order.order_id} after {max_retries} attempts")
                all_reconciled = False
        
        # Remove reconciled orders
        self.unreconciled_orders = [o for o in self.unreconciled_orders if not o.reconciled]
        self._save_state()
        
        return all_reconciled
    
    def get_unreconciled_count(self) -> int:
        """Get count of unreconciled orders"""
        return len([o for o in self.unreconciled_orders if not o.reconciled])
    
    def get_unreconciled_value(self) -> float:
        """Get total value of unreconciled orders"""
        return sum(o.fill_price * o.fill_size 
                   for o in self.unreconciled_orders if not o.reconciled)


def verify_order_status_on_kraken(kraken_connector, order_id: str) -> Optional[Dict]:
    """
    Verify order status directly with Kraken API
    
    Returns:
        Dict with fill details if order is filled, None otherwise
    """
    try:
        if kraken_connector.mode == 'paper':
            return None
        
        result = kraken_connector.api.query_private('QueryOrders', {'txid': order_id})
        
        if result.get('error'):
            logger.error(f"Error querying order: {result['error']}")
            return None
        
        orders = result.get('result', {})
        
        if order_id not in orders:
            logger.warning(f"Order {order_id} not found in Kraken")
            return None
        
        order_info = orders[order_id]
        status = order_info.get('status')
        
        if status == 'closed':
            return {
                'filled': True,
                'fill_price': float(order_info.get('price', 0)),
                'fill_size': float(order_info.get('vol_exec', 0)),
                'timestamp': datetime.now()
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error verifying order status: {e}")
        return None