import json
from pathlib import Path
import sys

# Añadir el root al path para poder importar si fuera necesario
sys.path.append(str(Path(__file__).parent.parent))

def audit_holdings():
    holdings_path = Path("config/holdings.txt")
    strategies_path = Path("config/holdings_strategies.json")
    
    print("=" * 50)
    print("AUDITORÍA DE ESTRATEGIAS (MARRIAGE CHECK)")
    print("=" * 50)

    # 1. Cargar Holdings
    if not holdings_path.exists():
        print(f"Error: No se encontró {holdings_path}")
        return

    with open(holdings_path, "r", encoding="utf-8-sig") as f:
        holdings = {line.split("#")[0].strip().upper() for line in f 
                    if line.strip() and not line.strip().startswith("#")}
    
    # 2. Cargar Estrategias Fijadas (Casados)
    married_dict = {}
    if strategies_path.exists():
        try:
            with open(strategies_path, "r", encoding="utf-8-sig") as f:
                married_dict = {k.upper(): v for k, v in json.load(f).items()}
        except Exception as e:
            print(f"Error leyendo JSON: {e}")

    married_tickers = set(married_dict.keys())
    
    # 3. Comparar
    unmarried = sorted(list(holdings - married_tickers))
    married_in_holdings = sorted(list(holdings & married_tickers))

    print(f"\n[CASADOS] ({len(married_in_holdings)} tickers con estrategia fija):")
    for t in married_in_holdings:
        print(f"  - {t}: {married_dict[t]}")

    print(f"\n[SOLTEROS / AUTO] ({len(unmarried)} tickers sin estrategia fija):")
    if not unmarried:
        print("  ¡Todos los holdings tienen una estrategia asignada!")
    for t in unmarried:
        print(f"  - {t} (usando selección automática)")

    print("\n" + "=" * 50)

if __name__ == "__main__":
    audit_holdings()