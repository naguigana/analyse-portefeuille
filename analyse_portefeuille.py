import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import os

# Crée le dossier 'images' s'il n'existe pas déjà
os.makedirs("images", exist_ok=True)

# Liste des tickers sélectionnés
tickers = [
    "APLD",  # Applied Digital
    "AMPY",  # Amplify Energy Corp
    "LPRO",  # Open Lending Corp
    "BRY",   # Berry Corp
    "ATNM",  # Actinium Pharmaceuticals
    "MKSI",  # MKS Instruments, Inc.
    "TSM",   # Taiwan Semiconductor
    "HIMX",  # Himax Technologies
    "NVAX",  # Novavax
    "ITOS"   # iTeos Therapeutics
]

def get_data(tickers, start, end):
    # On désactive auto_adjust pour garder la colonne 'Adj Close'
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    return data

def simulate_portfolios(returns, n_portfolios=5000, risk_free_rate=0.01):
    results = {'Returns': [], 'Volatility': [], 'Sharpe': [], 'Weights': []}
    for _ in range(n_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)

        port_return = np.sum(weights * returns.mean()) * 252
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility

        results['Returns'].append(port_return)
        results['Volatility'].append(port_volatility)
        results['Sharpe'].append(sharpe_ratio)
        results['Weights'].append(weights)

    return pd.DataFrame(results)

def plot_efficient_frontier(df):
    max_sharpe = df.iloc[df['Sharpe'].idxmax()]
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Volatility'], df['Returns'], c=df['Sharpe'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe['Volatility'], max_sharpe['Returns'], c='red', marker='*', s=200, label='Max Sharpe')
    plt.title('Frontière Efficiente - Optimisation de Portefeuille')
    plt.xlabel('Volatilité')
    plt.ylabel('Rendement attendu')
    plt.legend()
    plt.grid(True)

    # Sauvegarde du graphe (si dossier images/ existe)
    try:
        plt.savefig("images/portefeuille_optimal.png")
    except FileNotFoundError:
        print("💡 Le dossier 'images/' n'existe pas. Créé-le si tu veux sauvegarder le graphique.")
    plt.show()

def main():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3*365)

    print("📥 Téléchargement des données...")
    prices = get_data(tickers, start=start_date, end=end_date)
    returns = prices.pct_change().dropna()

    print("🧮 Simulation de portefeuilles...")
    portfolio_df = simulate_portfolios(returns)

    print("📈 Génération du graphe...")
    plot_efficient_frontier(portfolio_df)

    best = portfolio_df.iloc[portfolio_df['Sharpe'].idxmax()]
    allocation = pd.Series(best['Weights'], index=returns.columns)

    print("\n✅ Résultat de l'optimisation :")
    print("📊 Allocation optimale :\n", allocation.round(3))
    print("🎯 Rendement attendu :", round(best['Returns'], 3))
    print("📉 Volatilité :", round(best['Volatility'], 3))
    print("📈 Sharpe ratio :", round(best['Sharpe'], 3))

if __name__ == "__main__":
    main()
