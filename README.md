# Adaptive Split Inference of Transformer-Based NLP Models in Mobile Edge Computing Environments Using Reinforcement Learning

> **Исследовательский программный комплекс и симулятор распределённого адаптивного разделения инференса Transformer-моделей в средах мобильных граничных вычислений (MEC) с использованием обучения с подкреплением.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0%2B-green.svg)](https://gymnasium.farama.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey.svg)](LICENSE)

---

## Содержание
1. [Математическая модель системы](#-математическая-модель-системы)
   - [1.1. Модель вычислений и энергопотребления смартфона (CMOS DVFS)](#21-модель-вычислений-и-энергопотребления-смартфона-cmos-dvfs)
   - [1.2. Модель размера промежуточных данных $S(k, b, N)$](#22-модель-размера-промежуточных-данных-sk-b-n)
   - [1.3. Модель радиоканала и задержки передачи](#23-модель-радиоканала-и-задержки-передачи)
   - [1.4. Модель MEC-сервера и системы очередей (M/M/1)](#24-модель-mec-сервера-и-системы-очередей-mm1)
   - [1.5. Модель квантования промежуточных активаций](#25-модель-квантования-промежуточных-активаций)
   - [1.6. Оптимизационная целевая функция и Reward](#26-оптимизационная-целевая-функция-и-reward)
2. [Исследуемые алгоритмы принятия решений](#-исследуемые-алгоритмы-принятия-решений)
3. [Сетевые сценарии и профили](#-сетевые-сценарии-и-профили)
4. [Архитектура программного комплекса](#-архитектура-программного-комплекса)
5. [Установка и быстрый старт](#-установка-и-быстрый-старт)
6. [Интерактивный Streamlit Dashboard (6 страниц)](#-интерактивный-streamlit-dashboard-6-страниц)
7. [Запуск тестов и воспроизводимость экспериментов](#-запуск-тестов-и-воспроизводимость-экспериментов)
8. [Научные результаты и сравнительный анализ](#-научные-результаты-и-сравнительный-анализ)

---

## Математическая модель системы

### Модель вычислений и энергопотребления смартфона (CMOS DVFS)

Вычислительная мощность мобильного процессора описывается классической кубической моделью CMOS DVFS:
$$P_{\text{comp}}(f) = P_{\text{idle}} + \kappa \cdot f^3$$

где:
- $P_{\text{idle}}$ — базовая статическая мощность в режиме ожидания (Вт),
- $\kappa$ — технологический коэффициент динамического переключения транзисторов ($\text{Вт} / \text{ГГц}^3$),
- $f$ — тактовая частота CPU ($\text{ГГц}$).

Время локального выполнения $k$ слоёв при длине последовательности $N$:
$$T_{\text{device}}(k, f, N) = \begin{cases}
0, & \text{если } k = 0 \\
\left( T_{\text{emb}} + k \cdot T_{\text{layer,ref}} \cdot \frac{N}{N_{\text{ref}}} \right) \cdot \frac{f_{\text{ref}}}{f}, & \text{если } k > 0
\end{cases}$$

Энергопотребление мобильного устройства:
$$E_{\text{comp}} = P_{\text{comp}}(f) \cdot T_{\text{device}}$$
$$E_{\text{tx}} = P_{\text{tx}} \cdot T_{\text{tx\_data}}$$
$$E_{\text{mobile}} = E_{\text{comp}} + E_{\text{tx}}$$

*Примечание*: Мобильный радиопередатчик потребляет мощность $P_{\text{tx}}$ только во время физической передачи данных в эфир ($T_{\text{tx\_data}}$).

---

### Модель размера промежуточных данных $S(k, b, N)$

Размер полезной нагрузки, передаваемой по беспроводному каналу на срезной точке $k$:
$$S(k, b, N) = \begin{cases}
N \cdot 4\text{ байт}, & k = 0 \quad (\text{сырые индексы токенов}) \\
N \cdot d \cdot \frac{b}{8}\text{ байт}, & 0 < k < L \quad (\text{квантованный тензор скрытых состояний}) \\
0\text{ байт}, & k = L \quad (\text{локальный инференс, передача отсутствует})
\end{cases}$$

где $d$ — размерность скрытого представления ($d = 768$ для DistilBERT/BERT-base), $b$ — разрядность квантования.

---

### Модель радиоканала и задержки передачи

Задержка беспроводного канала строго разделена на две физические составляющие:
1. **Время передачи битов полезной нагрузки**:
   $$T_{\text{tx\_data}} = \frac{S(k, b, N) \cdot 8}{B(t) \cdot 10^6}\text{ сек}$$
2. **Задержка распространения радиосигнала и задержка сети**:
   $$T_{\text{prop}} = \frac{D_{\text{channel}}(t)}{1000}\text{ сек}$$

Итоговое время сетевой доставки:
$$T_{\text{transmission}} = T_{\text{tx\_data}} + T_{\text{prop}}$$

Канал $B(t)$ моделирует затухание по закону распространения радиоволн, затенение (Log-Normal Shadowing) и многолучевые рэлеевские замирания (Rayleigh Fading).

---

### Модель MEC-сервера и системы очередей (M/M/1)

MEC-сервер оснащён GPU-ускорителем с фактором масштабирования $\eta_{\text{accel}}$:
$$T_{\text{mec}}(k, N) = \begin{cases}
0, & \text{если } k = L \\
\frac{(L - k) \cdot T_{\text{server\_layer}} \cdot \frac{N}{N_{\text{ref}}} + T_{\text{head}}}{\eta_{\text{accel}}}, & \text{если } k < L
\end{cases}$$

Для моделирования конкурентной нагрузки сторонних задач используется модель теории массового обслуживания $M/M/1$:
$$T_{\text{queue}} = \begin{cases}
0, & \text{если } k = L \\
\frac{\psi_{\text{MEC}}(t)}{\mu \cdot (1 - \psi_{\text{MEC}}(t))}, & \text{если } k < L
\end{cases}$$

где $\psi_{\text{MEC}} \in [0, 0.95)$ — коэффициент загрузки сервера, $\mu$ — интенсивность обслуживания.

---

### Модель квантования промежуточных активаций

В модуле `models/quantization.py` реализовано реальное симметричное/асимметричное квантование тензоров PyTorch:
- **FP32** (32 бита): эталонное представление без искажений.
- **INT16** (16 бит): компрессия $2\times$, $\text{SNR} \approx 88\text{ дБ}$, $\text{CosSim} > 0.9999$.
- **INT8** (8 бит): компрессия $4\times$, $\text{SNR} \approx 40\text{ дБ}$, $\text{CosSim} > 0.999$.
- **INT4** (4 бита): физическая побитовая упаковка двух 4-битных нибблов в один байт `uint8`. Компрессия $8\times$, $\text{SNR} \approx 16\text{ дБ}$, $\text{CosSim} > 0.985$.

Метрики оценки качества:
$$\text{MSE} = \frac{1}{|X|} \sum (X - \hat{X})^2$$
$$\text{SNR (dB)} = 10 \cdot \log_{10}\left(\frac{\mathbb{E}[X^2]}{\text{MSE}}\right)$$
$$\text{Cosine Similarity} = \frac{\langle X, \hat{X} \rangle}{\|X\|_2 \cdot \|\hat{X}\|_2}$$
$$\text{Quality Score } Q \in [0, 1], \quad D = 1 - Q$$

---

### Оптимизационная целевая функция и Reward

Многокритериальная функция стоимости объединяет задержку, энергию и деградацию качества:
$$J = \alpha \cdot \frac{T_{\text{total}}}{T_{\text{ref}}} + \beta \cdot \frac{E_{\text{mobile}}}{E_{\text{ref}}} + \lambda \cdot D$$

Функция награды агента Reinforcement Learning:
$$R = -J - \Pi_{\text{SLA}} - \Pi_{\text{Quality}}$$

где штраф за нарушение дедлайна SLA ($T_{\text{total}} > \text{SLA}$):
$$\Pi_{\text{SLA}} = w_{\text{SLA}} \cdot \frac{T_{\text{total}} - T_{\text{SLA}}}{T_{\text{SLA}}}$$

и штраф за недопустимое падение точности ($Q < Q_{\min}$):
$$\Pi_{\text{Quality}} = w_Q \cdot (Q_{\min} - Q)$$

---

## Исследуемые алгоритмы принятия решений

В системе реализованы и верифицированы 8 политик управления:

| № | Алгоритм | Тип | Описание поведения |
|:---:|:---|:---:|:---|
| **1** | **Local Inference** | Базовый | Все слои вычисляются локально ($k = L$). Радиопередача отсутствует. Высокая нагрузка на батарею. |
| **2** | **Full Offloading** | Базовый | Полная выгрузка сырых токенов на сервер ($k = 0$). Минимальный расход энергии мобильного терминала, но сильная зависимость от сети. |
| **3** | **Fixed Split** | Базовый | Статическое разделение (по умолчанию $k = 3$, $b = 8$). Отсутствие динамической адаптации. |
| **4** | **Greedy Adaptive** | Эвристика | Аналитический перебор всех $(L+1) \times 4 \times N_f$ допустимых действий с минимизацией мгновенной функции $J$. |
| **5** | **Random Baseline** | Базовый | Случайный выбор допустимых действий для оценки базового уровня. |
| **6** | **DQN (Deep Q-Network)** | RL | Обучение Q-функции на общем дискретном пространстве действий (140 комбинаций) с Replay Buffer и Target-сетью. |
| **7** | **PPO (Proximal Policy Optimization)** | RL | Факторизованная мульти-дискретная сеть с 3 независимыми категориальными головами $(k, b, f)$, clipped surrogate objective и GAE. |
| **8** | **SAC (Soft Actor-Critic)** | RL | Дискретный Soft Actor-Critic с Twin Q-Critics, энтропийной регуляризацией $\mathcal{H}(\pi)$ и Polyak-обновлением target-сетей. |

---

## Сетевые сценарии и профили

Симулятор поддерживает 5 воспроизводимых сетевых сценариев (стабильное зерно случайности `seed`):

1. **`STABLE`**: стабильный 5G/Wi-Fi канал ($B \approx 50\text{ Мбит/с}$, $\text{Latency} \approx 10\text{ мс}$).
2. **`FLUCTUATING`**: периодические колебания и интерференция (10–70 Мбит/с).
3. **`CONGESTED`**: перегрузка базовой станции ($B \approx 5\text{ Мбит/с}$, высокая задержка в очереди $T_{\text{queue}}$).
4. **`POOR`**: слабое покрытие сотовой сети ($B \approx 1.5\text{ Мбит/с}$, $\text{Latency} \approx 95\text{ мс}$).
5. **`RANDOM`**: стохастическое блуждание параметров радиосреды.

---

## Архитектура программного комплекса

```
.
├── simulation/                 # Ядро симуляционной физической среды MEC
│   ├── device.py               # MobileDevice: CMOS DVFS, батарея, локальные вычисления
│   ├── mec_server.py           # MECServer: GPU-ускорение, M/M/1 очередь
│   ├── network.py              # NetworkChannel: 5 сценариев, Rayleigh fading, S(k,b,N)
│   ├── task_generator.py       # TaskGenerator: генератор запросов с воспроизводимым seed
│   ├── simpy_des.py            # SimPyDESEngine: дискретно-событийная очередь и ресурсы SimPy
│   ├── environment.py          # Gymnasium-совместимая среда MECSimulationEnv поверх SimPy DES
│   └── metrics.py              # Структуры данных StepMetrics, EpisodeMetrics, MetricsCollector
│
├── models/                     # Нейросетевые модели и квантование
│   ├── quantization.py         # ActivationQuantizer: FP32, INT16, INT8, INT4 с побитовой упаковкой
│   ├── transformer_model.py    # ModularTransformer: слои, эмбеддинги, послойное профилирование
│   └── split_inference.py      # SplitInferenceExecutor: раздельный инференс тензоров
│
├── algorithms/                 # 8 исследуемых алгоритмов
│   ├── base_policy.py          # Базовый абстрактный класс BasePolicy
│   ├── local_inference.py      # Алгоритм 1: Local Inference (k = L)
│   ├── full_offloading.py      # Алгоритм 2: Full Offloading (k = 0)
│   ├── fixed_split.py          # Алгоритм 3: Fixed Split (k = 3, b = 8)
│   ├── greedy.py               # Алгоритм 4: Greedy Adaptive Heuristic
│   ├── random_policy.py        # Алгоритм 5: Random Baseline
│   ├── dqn.py                  # Алгоритм 6: Deep Q-Network Policy
│   ├── ppo.py                  # Алгоритм 7: Proximal Policy Optimization (MultiDiscrete)
│   └── sac.py                  # Алгоритм 8: Soft Actor-Critic (Discrete Max Entropy)
│
├── training/                   # Пайплайн обучения и оценки RL
│   ├── train.py                # Цикл обучения RL-агентов
│   ├── evaluation.py           # Оценка обученных политик
│   └── model_manager.py        # Сохранение/загрузка чекпоинтов
│
├── experiments/                # Экспериментальный модуль
│   ├── experiment_runner.py    # ExperimentRunner: запуск серий опытов, сбор 10 метрик
│   ├── scenarios.py            # Пресеты параметров для 5 сетевых сценариев
│   └── result_storage.py       # Сохранение и парсинг результатов в CSV/JSON
│
├── dashboard/                  # Модули Streamlit Dashboard (6 страниц)
│   ├── overview.py             # Страница 1: Архитектура и живое состояние
│   ├── config_page.py          # Страница 2: Конфигурация параметров симуляции
│   ├── simulation_page.py      # Страница 3: Интерактивный симулятор разделения
│   ├── training_page.py        # Страница 4: Мониторинг обучения RL
│   ├── comparison_page.py      # Страница 5: Сравнение 8 алгоритмов и экспорт
│   └── analysis_page.py        # Страница 6: Аналитический научный отчёт
│
├── tests/                      # Набор тестов для верификации системы
│   ├── test_simulation.py      # Тесты физической среды и DVFS
│   ├── test_simpy_des.py       # Тесты дискретно-событийного движка SimPy и очередей
│   ├── test_stage2_transformer_quantization.py # Тесты квантования и моделей
│   ├── test_stage3_baseline_algorithms.py      # Тесты базовых политик
│   ├── test_stage4_rl_algorithms.py            # Тесты DQN, PPO, SAC
│   ├── test_stage5_experiments.py              # Тесты сборщика 10 метрик
│   └── test_stage7_integration.py              # Сквозные интеграционные тесты
│
├── app.py                      # Точка входа веб-интерфейса Streamlit
├── run_demo.py                 # Консольный бенчмарк алгоритмов
├── demo_scenario.py            # Демонстрация адаптации к деградации канала
└── run_tests.py                # Скрипт запуска полного набора тестов
```

---

## Установка и быстрый старт

### Требования
- Python 3.10+ или 3.11+
- Виртуальное окружение `venv` или `conda`

### 1. Клонирование и установка зависимостей
```bash
# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate  # Для Linux/macOS
# venv\Scripts\activate   # Для Windows

# Установка библиотек
pip install -r requirements.txt
# Или вручную:
# pip install torch gymnasium simpy streamlit plotly pandas numpy scipy scikit-learn
```

### 2. Запуск демонстрационного сценария (CLI)
Скрипт наглядно демонстрирует поведение моделей при резком ухудшении радиоканала:
```bash
python demo_scenario.py
```

### 3. Запуск Streamlit Dashboard
```bash
streamlit run app.py
```
После запуска панель управления откроется в браузере по адресу `http://localhost:8501`.

---

## Интерактивный Streamlit Dashboard

1. **Overview & Architecture** (`dashboard/overview.py`):
   - Графическая схема распределения вычислительных графов между Mobile Terminal и MEC Server.
   - Живые индикаторы параметров радиоканала, частоты CPU и фоновой загрузки серверов.
2. **Simulation Configuration** (`dashboard/config_page.py`):
   - Интерактивная настройка $L, d, N_{\text{tokens}}$, параметров CMOS ($\kappa, P_{\text{idle}}, P_{\text{tx}}$), коэффициента ускорения MEC и весов $\alpha, \beta, \lambda$.
3. **Live Simulation & Partitioning** (`dashboard/simulation_page.py`):
   - Визуализатор слоёв Transformer (зелёный — Mobile, фиолетовый — MEC, срез квантования).
   - Пошаговое выполнение инференса с детальной разбивкой времени и энергии.
4. **RL Agent Training** (`dashboard/training_page.py`):
   - Обучение агентов DQN, PPO, SAC в реальном времени.
   - Графики сходимости функции потерь (Critic Loss / Policy Loss), динамика наград и эволюция выбора точки среза $k$.
5. **Algorithm Comparison** (`dashboard/comparison_page.py`):
   - Параллельное тестирование всех 8 алгоритмов в идентичных условиях.
   - Построение гистограмм латентности, перцентилей P95, энергопотребления и коэффициента успешности SLA.
6. **Results & Scientific Analysis** (`dashboard/analysis_page.py`):
   - Построение фронта Парето («Задержка vs Энергия»).
   - Матрица относительного выигрыша адаптивного подхода относительно Local Inference и Full Offloading.
   - Экспорт таблиц результатов в формат CSV.

---

## Запуск тестов и воспроизводимость экспериментов

```bash
python run_tests.py
```

Пример успешного вывода проверки:
```
=====================================================================================
  MEC RL SPLIT INFERENCE SYSTEM - SCIENTIFIC VERIFICATION SUITE
  Testing Stages 1 through 7
=====================================================================================
[EXEC] Stage 1: MEC Environment & DVFS (tests/test_simulation.py)...             [PASS]
[EXEC] Stage 2: Transformer & Tensor Quantization (tests/test_stage2...)...      [PASS]
[EXEC] Stage 3: Baseline Algorithms (1-5) (tests/test_stage3...)...              [PASS]
[EXEC] Stage 4: RL Algorithms (DQN, PPO, SAC) (tests/test_stage4...)...          [PASS]
[EXEC] Stage 5: Experiment Runner & Metrics (tests/test_stage5...)...            [PASS]
[EXEC] Stage 7: End-to-End System Integration (tests/test_stage7...)...          [PASS]
=====================================================================================
  EXECUTION SUMMARY: 6 passed, 0 failed in 9.99s
=====================================================================================
SUCCESS: All scientific models, algorithms, and integration tests verified!
```
