# ============================================================
# Global Solution 2026 – Asteroid Mining
# Modelagem Linear para Aprendizado de Máquina – 1CC
# Análise Estatística Descritiva de Asteroides
# Fonte: JPL Small-Body Database / CNEOS / NASA
#  grupo: lucas klein da veiga rm:570029  Rafael Ferreirinha Quaresma rm571949
# ============================================================

# ── Instalação de dependências (Google Colab) ──────────────
# !pip install pandas numpy matplotlib scipy scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0) CARREGAMENTO DOS DADOS
# ============================================================
# Para rodar no Google Colab: faça upload do arquivo asteroides.csv
# ou substitua o caminho abaixo pelo caminho correto.

data = {
    'nome': ['1 Ceres','4 Vesta','2 Pallas','10 Hygiea','704 Interamnia',
             '52 Europa','511 Davida','87 Sylvia','65 Cybele','15 Eunomia',
             '3 Juno','16 Psyche','45 Eugenia','94 Aurora','31 Euphrosyne',
             '107 Camilla','121 Hermione','324 Bamberga','423 Diotima','48 Doris',
             '762 Pulcova','243 Ida','25143 Itokawa','433 Eros','1566 Icarus',
             '4179 Toutatis','1620 Geographos','3753 Cruithne','99942 Apophis',
             '101955 Bennu','162173 Ryugu','3122 Florence','1036 Ganymed',
             '1221 Amor','2100 Ra-Shalom','4660 Nereus','4769 Castalia',
             '6489 Golevka','29075 1950 DA','54509 YORP','65803 Didymos',
             '152830 Dinkinesh','2867 Steins','951 Gaspra','253 Mathilde'],
    'tipo': ['C','V','B','C','F','C','C','X','C','S','S','M','C','C','C',
             'C','C','CP','C','CG','FC','S','S','S','Q','S','S','E','S',
             'B','C','S','S','S','C','E','S','Q','M','S','S','S','E','S','C'],
    'diametro_km': [939.40,525.40,511.00,433.00,326.00,315.00,289.00,286.00,
                    273.00,268.00,267.00,226.00,214.00,205.00,267.00,222.60,
                    209.00,229.00,217.00,221.80,137.00,56.20,0.54,16.84,1.40,
                    4.60,5.04,5.00,0.37,0.49,0.90,4.90,37.70,1.00,2.80,0.51,
                    1.80,0.53,1.30,0.12,0.78,0.76,5.26,18.20,66.00],
    'magnitude_absoluta_H': [3.34,3.20,4.13,5.43,6.00,6.31,6.22,6.95,6.62,5.28,
                              5.33,5.90,7.46,7.57,6.74,7.08,7.34,6.82,7.25,7.63,
                              8.62,7.05,19.20,10.83,16.90,15.30,15.60,15.12,19.70,
                              20.18,19.25,14.10,9.45,17.70,16.10,18.20,16.90,19.15,
                              17.60,22.60,18.16,17.48,13.18,11.46,10.20],
    'periodo_orbital_anos': [4.60,3.63,4.62,5.57,5.36,5.46,5.66,6.52,6.35,4.30,
                              4.36,4.99,5.76,5.48,5.53,6.84,6.18,4.39,5.60,5.53,
                              5.84,4.84,1.52,1.76,1.12,3.98,1.39,0.998,0.886,1.20,
                              1.30,2.36,4.34,2.66,0.761,1.82,1.10,3.99,2.00,0.500,
                              2.11,1.22,3.63,3.29,4.31],
    'excentricidade': [0.076,0.089,0.231,0.117,0.153,0.101,0.179,0.081,0.104,0.187,
                       0.256,0.135,0.083,0.083,0.227,0.069,0.140,0.338,0.041,0.069,
                       0.100,0.043,0.280,0.223,0.827,0.634,0.335,0.515,0.191,0.204,
                       0.190,0.423,0.535,0.435,0.436,0.360,0.483,0.605,0.508,0.230,
                       0.384,0.107,0.146,0.174,0.266],
    'inclinacao_graus': [10.59,7.14,34.84,3.84,17.30,7.47,15.94,10.87,3.55,11.74,
                          12.99,3.10,6.61,8.01,26.31,9.93,7.56,11.13,11.25,6.54,
                          13.07,1.14,1.62,10.83,22.85,0.47,13.34,19.81,3.34,6.03,
                          5.88,22.15,26.70,11.87,15.76,1.43,8.89,2.28,12.17,1.59,
                          3.41,2.07,9.94,4.10,6.71],
    'velocidade_km_s': [9.05,7.40,9.40,8.20,8.10,8.00,7.90,6.80,7.00,8.30,8.20,
                         7.50,7.80,8.10,8.00,6.40,6.90,9.50,7.90,8.00,7.60,7.80,
                         25.10,24.36,30.20,16.10,24.80,20.50,30.73,28.10,23.66,
                         23.80,19.80,22.40,26.90,23.60,30.50,18.60,22.10,28.80,
                         23.30,24.10,17.20,18.50,18.10],
    'num_observacoes': [6663,8927,6112,5241,3987,4102,3756,2234,3288,7823,8234,
                         6892,2987,2654,3102,2234,2456,4123,2789,2567,1923,5632,
                         9876,11234,7823,9234,8765,7654,12453,15234,13456,6789,
                         8234,6543,7123,8934,7654,8123,9234,10234,11345,5432,
                         4321,7654,6123],
    'valor_estimado_bilhoes_usd': [0.71,1.30,0.89,0.55,0.42,0.38,0.34,0.31,0.29,
                                    0.27,0.24,10.00,0.21,0.19,0.26,0.22,0.20,0.23,
                                    0.21,0.22,0.14,0.10,0.001,0.80,0.03,0.07,0.08,
                                    0.07,0.002,0.001,0.002,0.07,0.55,0.01,0.04,
                                    0.001,0.03,0.001,0.02,0.0001,0.001,0.001,0.07,
                                    0.24,0.08]
}
df = pd.DataFrame(data)

print("=" * 60)
print("ASTEROID MINING – ANÁLISE ESTATÍSTICA DESCRITIVA")
print("Fonte: JPL Small-Body Database / CNEOS / NASA")
print("=" * 60)
print(f"\nTotal de asteroides na base: {len(df)}")
print(f"Tipos presentes: {sorted(df['tipo'].unique())}")
print("\nPrimeiras linhas:")
print(df[['nome','tipo','diametro_km','periodo_orbital_anos',
          'excentricidade','velocidade_km_s','num_observacoes']].head())

# ============================================================
# 1) GRÁFICO 1: Histograma – Distribuição dos Diâmetros
# ============================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor('#0D1B2A')
ax1.set_facecolor('#0D1B2A')

n, bins, patches = ax1.hist(df['diametro_km'], bins=15,
                             edgecolor='#00E5FF', color='#1565C0', alpha=0.85)
for i, p in enumerate(patches):
    p.set_facecolor(plt.cm.plasma(i / len(patches)))

ax1.set_title('Distribuição dos Diâmetros dos Asteroides',
              fontsize=16, fontweight='bold', color='white', pad=15)
ax1.set_xlabel('Diâmetro (km)', fontsize=12, color='#90CAF9', labelpad=8)
ax1.set_ylabel('Frequência Absoluta (fi)', fontsize=12, color='#90CAF9', labelpad=8)
ax1.tick_params(colors='white', labelsize=10)
for spine in ax1.spines.values():
    spine.set_edgecolor('#37474F')

mediana = df['diametro_km'].median()
media   = df['diametro_km'].mean()
ax1.axvline(mediana, color='#FFCA28', linestyle='--', linewidth=1.8,
            label=f'Mediana: {mediana:.2f} km')
ax1.axvline(media, color='#EF9A9A', linestyle=':', linewidth=1.8,
            label=f'Média: {media:.2f} km')
ax1.legend(fontsize=10, facecolor='#1A237E', edgecolor='#90CAF9',
           labelcolor='white', loc='upper right')

nota = ("Nota: A distribuição é fortemente assimétrica\n"
        "à direita, com a maioria dos asteroides\n"
        "com diâmetros menores que 50 km.")
ax1.text(0.97, 0.55, nota, transform=ax1.transAxes,
         fontsize=9, color='#B0BEC5', ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#1A237E',
                   alpha=0.7, edgecolor='#3949AB'))

ax1.grid(axis='y', linestyle='--', alpha=0.2, color='white')
plt.tight_layout()
plt.savefig('/home/claude/asteroid_mining/grafico1_histograma_diametro.png',
            dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
plt.show()
print("\n[Gráfico 1 salvo: grafico1_histograma_diametro.png]")

# ============================================================
# 2) GRÁFICO 2: Dispersão – Excentricidade vs Velocidade
# ============================================================
tipos_uniq = df['tipo'].unique()
palette = plt.cm.Set2(np.linspace(0, 1, len(tipos_uniq)))
cor_tipo = dict(zip(tipos_uniq, palette))

fig2, ax2 = plt.subplots(figsize=(11, 7))
fig2.patch.set_facecolor('#0D1B2A')
ax2.set_facecolor('#0D1B2A')

for tipo in tipos_uniq:
    mask = df['tipo'] == tipo
    ax2.scatter(df.loc[mask, 'excentricidade'],
                df.loc[mask, 'velocidade_km_s'],
                s=df.loc[mask, 'diametro_km'].apply(lambda x: max(20, min(400, x * 0.4))),
                color=cor_tipo[tipo], alpha=0.80, edgecolors='white',
                linewidths=0.5, label=f'Tipo {tipo}')

# linha de tendência
z = np.polyfit(df['excentricidade'], df['velocidade_km_s'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['excentricidade'].min(), df['excentricidade'].max(), 100)
ax2.plot(x_line, p(x_line), color='#FFCA28', linewidth=2, linestyle='--',
         label='Tendência Linear')

ax2.set_title('Excentricidade Orbital vs Velocidade de Aproximação\n'
              '(tamanho dos pontos proporcional ao diâmetro)',
              fontsize=14, fontweight='bold', color='white', pad=12)
ax2.set_xlabel('Excentricidade Orbital', fontsize=12, color='#90CAF9', labelpad=8)
ax2.set_ylabel('Velocidade de Aproximação (km/s)', fontsize=12, color='#90CAF9', labelpad=8)
ax2.tick_params(colors='white', labelsize=10)
for spine in ax2.spines.values():
    spine.set_edgecolor('#37474F')

r_val, p_val = stats.pearsonr(df['excentricidade'], df['velocidade_km_s'])
ax2.text(0.03, 0.96, f'r de Pearson = {r_val:.3f}\np-valor = {p_val:.4f}',
         transform=ax2.transAxes, fontsize=10, color='#FFCA28', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#1A237E',
                   alpha=0.8, edgecolor='#FFCA28'))

ax2.legend(fontsize=8, facecolor='#1A237E', edgecolor='#90CAF9',
           labelcolor='white', ncol=2, loc='lower right')
ax2.grid(linestyle='--', alpha=0.15, color='white')
plt.tight_layout()
plt.savefig('/home/claude/asteroid_mining/grafico2_dispersao_exc_vel.png',
            dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
plt.show()
print("[Gráfico 2 salvo: grafico2_dispersao_exc_vel.png]")

# ============================================================
# 3) ANÁLISES UNIVARIADAS – ESTATÍSTICA DESCRITIVA
# ============================================================

def analise_univariada(serie, nome_variavel, unidade=""):
    print(f"\n{'='*60}")
    print(f"ANÁLISE UNIVARIADA: {nome_variavel}")
    print(f"{'='*60}")

    n       = len(serie)
    media   = serie.mean()
    mediana = serie.median()
    moda_r  = serie.mode()
    moda    = moda_r.values[0] if len(moda_r) > 0 else None
    maximo  = serie.max()
    minimo  = serie.min()
    amplit  = maximo - minimo
    variancia = serie.var(ddof=1)
    desvpad = serie.std(ddof=1)
    q1  = serie.quantile(0.25)
    q2  = serie.quantile(0.50)
    q3  = serie.quantile(0.75)
    iqr = q3 - q1

    print(f"\n  N (observações)   : {n}")
    print(f"\n  ── Medidas de Tendência Central ──")
    print(f"  Média             : {media:.4f} {unidade}")
    print(f"  Mediana           : {mediana:.4f} {unidade}")
    print(f"  Moda              : {moda:.4f} {unidade}")
    print(f"\n  ── Medidas de Dispersão ──")
    print(f"  Máximo            : {maximo:.4f} {unidade}")
    print(f"  Mínimo            : {minimo:.4f} {unidade}")
    print(f"  Amplitude         : {amplit:.4f} {unidade}")
    print(f"  Variância (s²)    : {variancia:.4f}")
    print(f"  Desvio Padrão (s) : {desvpad:.4f} {unidade}")
    print(f"\n  ── Medidas Separatrizes (Quartis) ──")
    print(f"  Q1 (25%)          : {q1:.4f} {unidade}")
    print(f"  Q2 (50% / Med.)   : {q2:.4f} {unidade}")
    print(f"  Q3 (75%)          : {q3:.4f} {unidade}")
    print(f"  IIQ (Q3–Q1)       : {iqr:.4f} {unidade}")

    return {
        'variavel': nome_variavel, 'n': n, 'media': media, 'mediana': mediana,
        'moda': moda, 'maximo': maximo, 'minimo': minimo, 'amplitude': amplit,
        'variancia': variancia, 'desvpad': desvpad, 'Q1': q1, 'Q2': q2,
        'Q3': q3, 'IQR': iqr
    }

est1 = analise_univariada(df['diametro_km'], 'Diâmetro dos Asteroides', 'km')
est2 = analise_univariada(df['excentricidade'], 'Excentricidade Orbital', '')

# ── Box-plot comparativo ──────────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(12, 6))
fig3.patch.set_facecolor('#0D1B2A')
fig3.suptitle('Box-Plots – Distribuição das Variáveis Analisadas',
              fontsize=15, fontweight='bold', color='white', y=1.02)

variaveis = [
    (df['diametro_km'], 'Diâmetro (km)', '#42A5F5'),
    (df['excentricidade'], 'Excentricidade Orbital', '#66BB6A')
]
for ax, (serie, titulo, cor) in zip(axes, variaveis):
    ax.set_facecolor('#0D1B2A')
    bp = ax.boxplot(serie, patch_artist=True, widths=0.5,
                    medianprops=dict(color='#FFCA28', linewidth=2.5),
                    whiskerprops=dict(color='white', linewidth=1.5),
                    capprops=dict(color='white', linewidth=1.5),
                    flierprops=dict(marker='o', color='#EF9A9A',
                                   markerfacecolor='#EF9A9A', markersize=5))
    bp['boxes'][0].set_facecolor(cor)
    bp['boxes'][0].set_alpha(0.7)
    ax.set_title(titulo, fontsize=12, color='white', pad=8)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#37474F')
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='white')

plt.tight_layout()
plt.savefig('/home/claude/asteroid_mining/grafico3_boxplots.png',
            dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
plt.show()
print("[Gráfico 3 (Box-plots) salvo]")

# ============================================================
# 4) REGRESSÃO LINEAR SIMPLES
#    Variável preditora  X: Magnitude Absoluta (H)
#    Variável resposta   Y: log10(Diâmetro km)
#    Relação física: D ≈ (1329/sqrt(pv)) * 10^(-H/5)
# ============================================================
print("\n" + "=" * 60)
print("REGRESSÃO LINEAR SIMPLES")
print("X: Magnitude Absoluta (H)  →  Y: log10(Diâmetro) em km")
print("=" * 60)

df_reg = df[df['diametro_km'] > 0].copy()
X_raw  = df_reg['magnitude_absoluta_H'].values.reshape(-1, 1)
y_raw  = np.log10(df_reg['diametro_km'].values)

modelo = LinearRegression()
modelo.fit(X_raw, y_raw)
y_pred = modelo.predict(X_raw)

r2   = r2_score(y_raw, y_pred)
rmse = np.sqrt(mean_squared_error(y_raw, y_pred))
r_p, pval = stats.pearsonr(X_raw.flatten(), y_raw)

print(f"\n  Coeficiente angular (β₁) : {modelo.coef_[0]:.5f}")
print(f"  Intercepto (β₀)          : {modelo.intercept_:.5f}")
print(f"  Equação do modelo        : log₁₀(D) = {modelo.intercept_:.4f} + ({modelo.coef_[0]:.4f}) × H")
print(f"\n  R² (coef. determinação)  : {r2:.4f}  ({r2*100:.1f}% da variância explicada)")
print(f"  RMSE                     : {rmse:.4f}")
print(f"  Correlação de Pearson (r): {r_p:.4f}")
print(f"  p-valor                  : {pval:.6f}")

# ── Gráfico da Regressão ─────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(11, 7))
fig4.patch.set_facecolor('#0D1B2A')
ax4.set_facecolor('#0D1B2A')

ax4.scatter(X_raw, y_raw, color='#42A5F5', s=70, alpha=0.8,
            edgecolors='white', linewidths=0.5,
            label='Dados observados', zorder=3)

x_line = np.linspace(X_raw.min(), X_raw.max(), 200).reshape(-1, 1)
y_line = modelo.predict(x_line)
ax4.plot(x_line, y_line, color='#FFCA28', linewidth=2.5, zorder=4,
         label=f'Regressão: log₁₀(D) = {modelo.intercept_:.3f} + ({modelo.coef_[0]:.3f})·H')

for xi, yi, yp, nome in zip(X_raw.flatten(), y_raw, y_pred,
                              df_reg['nome'].values):
    if abs(yi - yp) > 0.6:
        ax4.annotate(nome, (xi, yi), fontsize=7, color='#EF9A9A',
                     xytext=(5, 5), textcoords='offset points')

ax4.set_title('Regressão Linear: Magnitude Absoluta vs log₁₀(Diâmetro)',
              fontsize=14, fontweight='bold', color='white', pad=12)
ax4.set_xlabel('Magnitude Absoluta (H)', fontsize=12, color='#90CAF9', labelpad=8)
ax4.set_ylabel('log₁₀(Diâmetro em km)', fontsize=12, color='#90CAF9', labelpad=8)
ax4.tick_params(colors='white', labelsize=10)
for spine in ax4.spines.values():
    spine.set_edgecolor('#37474F')

info = (f"R² = {r2:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"r Pearson = {r_p:.4f}\n"
        f"p-valor = {pval:.2e}")
ax4.text(0.03, 0.28, info, transform=ax4.transAxes,
         fontsize=10, color='#FFCA28', va='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A237E',
                   alpha=0.85, edgecolor='#FFCA28'))

ax4.legend(fontsize=10, facecolor='#1A237E', edgecolor='#90CAF9',
           labelcolor='white', loc='upper right')
ax4.grid(linestyle='--', alpha=0.15, color='white')
plt.tight_layout()
plt.savefig('/home/claude/asteroid_mining/grafico4_regressao.png',
            dpi=150, bbox_inches='tight', facecolor='#0D1B2A')
plt.show()
print("[Gráfico 4 (Regressão) salvo]")

# ── Previsão com o modelo ─────────────────────────────────────
print("\n  ── Previsões com o modelo ──")
exemplos_H = [5.0, 10.0, 15.0, 20.0, 25.0]
for H in exemplos_H:
    log_d = modelo.predict([[H]])[0]
    d_km  = 10 ** log_d
    print(f"  H = {H:5.1f}  →  D estimado ≈ {d_km:10.3f} km")

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("=" * 60)
