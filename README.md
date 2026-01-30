# ⚡ Energy-Aware Integrators — RK4 vs Symplectic vs ML

> **Simulações físicas que respeitam a energia. Machine Learning que entende a física.**

Este projeto é um **experimento científico + open source colaborativo** que compara **integradores numéricos clássicos**, **integradores simpléticos** e **modelos de Machine Learning** em sistemas dinâmicos sensíveis (caóticos / Hamiltonianos), com foco especial em **conservação de energia no longo prazo**.

Aqui, não basta o modelo "funcionar" — ele precisa **respeitar as leis da física**.

---

## 🚀 Motivação

Métodos clássicos como **RK4** são amplamente usados, mas:

* ❌ Não preservam invariantes físicos (energia, momento)
* ❌ Acumulam erro exponencial em sistemas caóticos
* ❌ Podem produzir resultados **não físicos** após certo tempo

Por outro lado:

* 🔁 Integradores **simpléticos** (ex: Velocity Verlet) preservam a estrutura Hamiltoniana
* 🤖 Modelos de **Machine Learning físico-informados** podem aprender o *fluxo do sistema*, não apenas a derivada

Este projeto investiga:

> **ML pode ser tão bom (ou melhor) que integradores clássicos na conservação de energia?**

Spoiler: os resultados são bem interessantes.

---

## 🧠 O que este projeto faz

* Implementa e compara:

  * RK4 (Runge-Kutta de 4ª ordem)
  * Velocity Verlet (simplético)
  * Modelos de ML (PyTorch)
  * Integração híbrida (Residual ML + integrador físico)

* Avalia:

  * Conservação de energia
  * Estabilidade numérica
  * Erro acumulado no tempo

* Visualiza:

  * Energia total vs tempo
  * Divergência entre métodos

---

## 📊 Exemplo de resultado

Em sistemas sensíveis, observamos:

* 🔵 **RK4** explodindo energia (instabilidade numérica)
* 🟢 **Verlet** mantendo energia estável
* 🟠 **ML** aprendendo o manifold energético

> Em alguns cenários, o ML preserva energia melhor que RK4 clássico.

---

## 🗂️ Estrutura do projeto

```text
.
├── integrators/
│   ├── rk4.py
│   ├── verlet.py
│   └── compare_integrators.py
│
├── ml/
│   ├── models/
│   │   └── energy_net.py
│   ├── train_ml.py
│   └── energy_torch.py
│
├── physics/
│   ├── system.py
│   └── energy.py
│
├── experiments/
│   └── run_experiment.py
│
├── plots/
│   └── energy_comparison.png
│
└── README.md
```

---

## 🛠️ Tecnologias

* **Python 3.10+**
* **NumPy**
* **PyTorch**
* **Matplotlib**

Sem dependências obscuras. Fácil de rodar, fácil de contribuir.

---

## ▶️ Como rodar

```bash
# clone o projeto
git clone https://github.com/seu-usuario/energy-aware-integrators.git
cd energy-aware-integrators

# instale dependências
pip install -r requirements.txt

# execute um experimento
python experiments/run_experiment.py
```

---

## 🤝 Projeto colaborativo

Este é um **projeto aberto e colaborativo**.

Você pode contribuir com:

* 📈 Novos sistemas físicos (pêndulo duplo, N-body, órbitas)
* 🤖 Novas arquiteturas de ML (Hamiltonian NN, Symplectic NN)
* ⚙️ Novos integradores
* 📊 Métricas e visualizações
* 🧪 Experimentos e benchmarks
* 📚 Documentação e explicações

### Como contribuir

1. Faça um fork
2. Crie uma branch (`feature/minha-ideia`)
3. Commit com mensagem clara
4. Abra um Pull Request 🚀

Toda contribuição é bem-vinda — do iniciante ao pesquisador.

---

## 📐 Diretrizes

* Código limpo e legível
* Comentários explicando *o porquê*, não só *o quê*
* Resultados devem ser reprodutíveis
* ML **não deve violar leis físicas básicas** sem justificativa

---

## 🧪 Próximos passos (roadmap)

* [ ] Residual ML + Verlet
* [ ] Hamiltonian Neural Networks (HNN)
* [ ] Comparação com integradores simpléticos de ordem superior
* [ ] Benchmark em sistemas caóticos reais
* [ ] Escrita de artigo técnico

---

## 📜 Licença

MIT — use, modifique, experimente.

Se este projeto te ajudou, ⭐ o repositório.

---

## 🧠 Filosofia do projeto

> *"Não adianta prever o futuro se você quebra as leis do universo no caminho."*

Vamos construir simuladores mais inteligentes — **e mais físicos**.

---

Feito com ciência, curiosidade e código limpo 🧪⚙️
