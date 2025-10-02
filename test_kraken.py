# test_kraken.py
from src.data.kraken_connector import KrakenConnector

try:
    kraken = KrakenConnector()
    print(f"Kraken object created: {kraken}")
    print(f"Has api attribute: {hasattr(kraken, 'api')}")
    print(f"API object: {kraken.api}")
    
    if kraken.api:
        print(" Kraken connector initialized correctly!")
    else:
        print(" Kraken connector's api is None")
        
except Exception as e:
    print(f" Error creating Kraken connector: {e}")