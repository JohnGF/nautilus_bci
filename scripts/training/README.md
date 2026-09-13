# BCI Tower Defense — Rhythm Decoding Training Studio & Real-Time Bridge

Esta pasta (`nautilus_bci/scripts/Training`) contém a suíte completa de treino de modelos para descodificação de ritmos mentais (4 classes: **FIRE**, **WATER**, **WIND**, **ELECTRICITY**) no jogo **Tower Defense BCI**.

Permite selecionar os datasets BIDS (com diferentes músicas e salas de gravação), escolher os sujeitos e sessões a incluir no treino, treinar com qualquer um dos 10 algoritmos de descodificação com validação cruzada estratificada (5-fold CV), guardar os artefactos do modelo e iniciar imediatamente a pipeline de tempo real ligada ao Godot.

---

## 🖥️ Como Iniciar a Aplicação Gráfica (GUI)

Para abrir a interface gráfica interativa:

```bash
# A partir do diretório raiz do nautilus_bci ou da pasta Training:
uv run python scripts/Training/train_and_run.py
```
*(ou diretamente)*:
```bash
uv run python scripts/Training/gui.py
```

### Funcionalidades da Interface Gráfica:
1. **Seleção de Dataset & Variação Musical**:
   - `bids_tower_defense`: Sessões padrão e modernas (sub-02 com rock/pop: Kiss, What's Up, Thunderstruck, etc.; sub-01 com peças clássicas e variante de água).
   - `bids_tower_defense(old)`: Sessões originais com música clássica pura (Für Elise, Bach Prelude, Vivaldi Spring, Tchaikovsky Waltz).
   - `bids_tower_defense_6_3_27`: Variação de gravação individual (ses-01).
2. **Seleção de Sujeito e Sessões**:
   - Deteção automática dos sujeitos (`sub-01`, `sub-02`).
   - Checkboxes individuais por sessão (`ses-01`, `ses-02`, etc.) com contagem prévia de trials detetados.
   - Botões de conveniência *"Select All"* e *"Clear All"*.
3. **10 Algoritmos de Descodificação**:
   - Dropdown com todos os algoritmos avaliados em `analyze_tower_defense_rhythm_decoding.py`.
   - Opção especial **"⚡ Benchmark All"**: avalia todos os 10 modelos em 5-Fold CV e exporta automaticamente o que obtiver melhor acurácia!
4. **Painel de Métricas em Tempo Real**:
   - O treino corre numa thread secundária sem bloquear a interface.
   - Cartões de métricas: **Acurácia CV**, **Macro F1**, **Kappa de Cohen**, e **Total de Trials**.
   - Matriz de confusão e relatório detalhado na consola de texto.
5. **Ponte para Tempo Real com o Godot**:
   - Botão **"🚀 LAUNCH REAL-TIME GAME PIPELINE"** para iniciar a pipeline.
   - Escolha da fonte: **Simulador (Replay BIDS)**, **Simulador (Sintético)**, ou **Live LSL (g.Nautilus)**.
   - Limiar de ativação configurável (*confidence threshold*).
   - Envio de comandos UDP diretamente para o Godot (`127.0.0.1:4242`).

---

## ⚡ Utilização por Linha de Comandos (CLI)

Também podes correr o treino de forma headless ou automatizada via terminal:

### Exemplos:

#### 1. Treinar com Riemannian Tangent Space numa única sessão:
```bash
uv run python scripts/Training/train_and_run.py --dataset bids_tower_defense --sub 02 --ses 05 --alg riemann_logreg
```

#### 2. Treinar acumulando múltiplas sessões (ex: todas as sessões do sub-02):
```bash
uv run python scripts/Training/train_and_run.py --dataset bids_tower_defense --sub 02 --ses 01,02,03,04,05 --alg riemann_logreg
```

#### 3. Treinar com o dataset antigo (músicas clássicas originais):
```bash
uv run python scripts/Training/train_and_run.py --dataset "bids_tower_defense(old)" --sub 01 --ses 01,02,03 --alg fbcsp_logreg
```

#### 4. Correr Benchmark de todos os algoritmos e selecionar o melhor:
```bash
uv run python scripts/Training/train_and_run.py --dataset bids_tower_defense --sub 02 --ses all --alg all
```

#### 5. Treinar e arrancar logo a pipeline em tempo real com o simulador:
```bash
uv run python scripts/Training/train_and_run.py --dataset bids_tower_defense --sub 02 --ses 05 --alg riemann_logreg --run-pipeline --source simulator --mode bids_replay
```

---

## 🧠 Algoritmos Disponíveis

| Chave (`--alg`) | Nome | Descrição |
| :--- | :--- | :--- |
| `riemann_logreg` | Riemannian Tangent Space + LogReg | Geometria Riemanniana em variedades SPD (Fréchet Mean) + Regressão Logística L2 |
| `fbcsp_logreg` | Filter Bank CSP + LogReg | Filtragem espacial multi-banda (Theta, Alpha, Low/High Beta, Gamma) + LogReg |
| `csp_lda` | CSP + Shrinkage LDA | Filtros espaciais CSP One-vs-Rest com LDA regularizado (Ledoit-Wolf shrinkage) |
| `csp_svm` | CSP + SVM (RBF) | CSP combinado com máquina de vetores de suporte não-linear de base radial |
| `riemann_svm_lin`| Riemannian Tangent Space + Linear SVM | Tangent Space com Linear SVM e calibração de probabilidades (Platt scaling) |
| `riemann_ridge` | Riemannian Tangent Space + Ridge | Tangent Space com classificador Ridge L2 e calibração de probabilidades |
| `riemann_svm_rbf`| Riemannian Tangent Space + RBF SVM | Tangent Space com kernel RBF para fronteiras de decisão curvas |
| `welch_rf` | Welch PSD + Random Forest | Densidade espetral de potência em bandas clássicas EEG + Random Forest |
| `welch_lda` | Welch PSD + Shrinkage LDA | Bandpower Welch relativo + Linear Discriminant Analysis |
| `ensemble_voting`| Ensemble Soft Voting | Votação ponderada/suave entre CSP+LDA, FBCSP e Riemannian Tangent Space |
| `all` | Benchmark Competitivo | Avalia todos os 10 modelos e treina o melhor no dataset completo |

---

## 📂 Ficheiros e Estrutura Gerada

- `models/`: Pasta onde são guardados os artefactos gerados:
  - `rhythm_model_<tag>.joblib`: Modelo treinado, imediatamente pronto para ser consumido pelo `tower-defense-bci`.
  - `rhythm_report_<tag>.json`: Relatório completo com acurácia de validação cruzada, desvio-padrão, macro F1, kappa de Cohen e matriz de confusão.
- `algorithms.py`: Implementação dos estimadores scikit-learn compatíveis com inferência real-time por janela `(32, 750)`.
- `dataset.py`: Localizador de pastas BIDS, catálogo de sujeitos/sessões e rotinas de pré-processamento/extração de épocas.
- `gui.py`: Interface gráfica de desktop moderna com Tkinter.
- `train_and_run.py`: Ponto de entrada unificado para GUI e CLI.
