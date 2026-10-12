import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# PARTE 1: Generazione dei Dati Sintetici
# ==========================================
print("=== PARTE 1: Generazione Dati ===")

# Liste di base per negozi e prodotti italiani
negozi = ["Milano", "Roma", "Napoli", "Torino", 
         "Firenze", "Bologna", "Palermo", "Verona", "Bari", "Catania"]

prodotti = ["Smartphone", "Laptop", "Smart TV", "Tablet", "Smartwatch", 
            "Cuffie Wireless", "Console", "Fotocamera", "Monitor", "Stampante"]

# Generazione casuale di 30 righe di dati con NumPy
negozi_casuali = np.random.choice(negozi, size=30)
prodotti_casuali = np.random.choice(prodotti, size=30)
quantita = np.random.randint(1, 11, size=30) # Quantità tra 1 e 10 pezzi

# Prezzi unitari casuali tra 49.99€ e 1299.99€, arrotondati a 2 decimali
prezzi_unitari = np.random.uniform(49.99, 1299.99, size=30)
prezzi_unitari = np.round(prezzi_unitari, 2)

# Generazione di una sequenza temporale di 30 giorni consecutivi
date = pd.date_range(start="2023-09-01", periods=30).strftime('%Y-%m-%d').tolist()


# ==========================================
# PARTE 2: Importazione e Controllo con Pandas
# ==========================================
print("\n=== PARTE 2: Creazione e Controllo DataFrame ===")

# Creazione del dizionario e del DataFrame principale
dati = {
    "Negozi": negozi_casuali,
    "Prodotti": prodotti_casuali,
    "Quantita": quantita,
    "Prezzi_Unitari": prezzi_unitari,
    "Data": date
}

df_vendite = pd.DataFrame(dati)

# Salvataggio del file CSV iniziale simulando un dataset esterno
df_vendite.to_csv("vendite.csv", index=False)

# Controlli strutturali preliminari sul DataFrame
print("-> Prime 5 righe del dataset:\n", df_vendite.head(5))
print("\n-> Dimensioni (Righe, Colonne):", df_vendite.shape)
print("\n-> Informazioni generali sul dataset:")
print(df_vendite.info())


# ==========================================
# PARTE 3: Elaborazioni e Aggregazioni
# ==========================================
print("\n=== PARTE 3: Elaborazioni con Pandas ===")

# 1. Calcolo e aggiunta della colonna Incasso (Quantità * Prezzo Unitario)
df_vendite["Incasso"] = df_vendite["Quantita"] * df_vendite["Prezzi_Unitari"]

# 2. Calcoli statistici e raggruppamenti chiave
incasso_totale = df_vendite["Incasso"].sum()
incasso_medio_negozio = df_vendite.groupby("Negozi")["Incasso"].mean()
top3_prodotti_qta = df_vendite.groupby("Prodotti")["Quantita"].sum().nlargest(3)
incasso_medio_negozio_prodotto = df_vendite.groupby(["Negozi", "Prodotti"])["Incasso"].mean()

print(f"-> Incasso Totale della catena: €{incasso_totale:,.2f}")


# ==========================================
# PARTE 4: Analisi Statistica Avanzata (NumPy)
# ==========================================
print("\n=== PARTE 4: Statistiche con NumPy ===")

# Estrazione della colonna Quantità come array NumPy nativo
array_numpy_quantita = np.array(df_vendite["Quantita"])

media_qta = np.mean(array_numpy_quantita)
min_qta = np.min(array_numpy_quantita)
max_qta = np.max(array_numpy_quantita)
std_qta = np.std(array_numpy_quantita)
perc_sopra_media = (np.sum(array_numpy_quantita > media_qta) / len(array_numpy_quantita)) * 100

print(f"-> Media Quantità: {media_qta:.2f} (Min: {min_qta}, Max: {max_qta}, Std: {std_qta:.2f})")
print(f"-> Percentuale vendite sopra la media: {perc_sopra_media:.1f}%")

# Verifica incassi tramite array 2D di NumPy confrontati con il DataFrame
array_2d = np.column_stack((df_vendite["Quantita"], df_vendite["Prezzi_Unitari"]))
incasso_calcolato_np = array_2d[:, 0] * array_2d[:, 1]
print("\n-> Confronto calcolo NumPy 2D vs DataFrame completato con successo.")


# ==========================================
# PARTE 5: Visualizzazioni Grafiche (Matplotlib)
# ==========================================
print("\n=== PARTE 5: Generazione Grafici ===")

# 1. Grafico a barre: Incasso Totale per Negozio
df_vendite.groupby("Negozi")["Incasso"].sum().plot(kind="bar", color="skyblue")
plt.title("Incasso Totale per Negozio")
plt.xlabel("Negozio")
plt.ylabel("Incasso (€)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. Grafico a torta: Percentuale Incassi per Prodotto (con Legenda esterna pulita)
df_vendite.groupby("Prodotti")["Incasso"].sum().plot(
    kind="pie", 
    autopct="%1.1f%%", 
    pctdistance=0.85, 
    legend=False
)
plt.title("Percentuale Incassi per Prodotto")
plt.ylabel("")
plt.legend(title="Prodotti", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# 3. Grafico a linee: Andamento Giornaliero degli Incassi
incasso_giornaliero = df_vendite.groupby("Data")["Incasso"].sum()
incasso_giornaliero.plot(kind="line", marker='o', color="orange")
plt.title("Andamento Giornaliero Incassi")
plt.xlabel("Data")
plt.ylabel("Incasso (€)")
plt.xticks(ticks=range(len(incasso_giornaliero)), labels=incasso_giornaliero.index, rotation=45, ha='right')
plt.tight_layout()
plt.show()


# ==========================================
# PARTE 6: Analisi Avanzata e Mappatura Categorie
# ==========================================
print("\n=== PARTE 6: Mappatura Categorie e Salvataggio ===")

# Dizionario per raggruppare i prodotti in macro-famiglie
mappa_categorie = {
    "Smartphone": "Informatica",
    "Laptop": "Informatica",
    "Tablet": "Informatica",
    "Smart TV": "Audio/Video",
    "Monitor": "Audio/Video",
    "Smartwatch": "Accessori Tech",
    "Cuffie Wireless": "Accessori Tech",
    "Console": "Intrattenimento & Ufficio",
    "Fotocamera": "Intrattenimento & Ufficio",
    "Stampante": "Intrattenimento & Ufficio"
}

# Applicazione della mappatura tramite .map() di Pandas
df_vendite["Categoria"] = df_vendite["Prodotti"].map(mappa_categorie)
print("-> Anteprima prodotti mappati:\n", df_vendite[["Prodotti", "Categoria"]].head())

# Aggiornamenti statistici per categoria
incasso_totale_categoria = df_vendite.groupby("Categoria")["Incasso"].sum()
quantita_media_categoria = df_vendite.groupby("Categoria")["Quantita"].mean()

# Salvataggio del DataFrame finale arricchito con la nuova colonna
df_vendite.to_csv("vendite_analizzate.csv", index=False)
print("-> File finale salvato con successo come 'vendite_analizzate.csv'.")


# ==========================================
# PARTE 7: Estensioni e Grafico Combinato Avanzato
# ==========================================
print("\n=== PARTE 7: Estensioni (Per i più bravi) ===")

# Funzione riutilizzabile per estrodre i primi N prodotti per incasso totale
def top_n_prodotti(n):
    return df_vendite.groupby("Prodotti")["Incasso"].sum().nlargest(n)

print(f"-> Top 3 prodotti per incasso:\n{top_n_prodotti(3)}")

# Preparazione dati per il grafico combinato a doppio asse Y
incasso_medio_cat = df_vendite.groupby("Categoria")["Incasso"].mean()
quantita_media_cat = df_vendite.groupby("Categoria")["Quantita"].mean()

fig, ax1 = plt.subplots(figsize=(10, 6))

# Asse Y sinistro: Grafico a barre per l'incasso medio
ax1.bar(incasso_medio_cat.index, incasso_medio_cat.values, color='skyblue', alpha=0.8, label='Incasso Medio')
ax1.set_xlabel('Categoria')
ax1.set_ylabel('Incasso Medio (€', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
plt.xticks(rotation=45, ha='right')

# Asse Y destro: Grafico a linee per la quantità media venduta
ax2 = ax1.twinx()
ax2.plot(quantita_media_cat.index, quantita_media_cat.values, color='darkorange', marker='o', linewidth=2.5, label='Quantità Media')
ax2.set_ylabel('Quantità Media Venduta', color='darkorange')
ax2.tick_params(axis='y', labelcolor='darkorange')

plt.title('Confronto Avanzato: Incasso Medio (Barre) vs Quantità Media (Linea) per Categoria')
plt.tight_layout()
plt.show()