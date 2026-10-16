import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 1. КОНФИГУРАЦИЯ И СТРУКТУРА ДАННЫХ
# ==========================================
st.set_page_config(page_title="BKT Graph Adaptive Testing", layout="wide")

SKILLS = [
    "Арифметические операции",      # 0
    "Общий знаменатель",            # 1
    "Десятичные дроби",             # 2
    "Геометрические свойства",       # 3
    "Обыкновенные дроби",           # 4
    "Проценты от числа",            # 5
    "Периметр и площадь",           # 6
    "Логические закономерности",    # 7
    "Пропорции",                    # 8
    "Линейные уравнения",           # 9
    "Принцип Дирихле",              # 10
    "Текстовые задачи"              # 11
]

EDGES = [
    (0, 1), (0, 2), (0, 6),
    (1, 4),
    (2, 5), (2, 9),
    (3, 6),
    (4, 8),
    (5, 8),
    (7, 8), (7, 10),
    (8, 11), (9, 11)
]

MASTERY_THRESHOLD = 0.85
EPS_0, EPS_1 = 0.30, 0.05
RHO_0, RHO_1 = 0.60, 0.90
SLIP_DEFAULT = 0.10
TRANS_DEFAULT = 0.20

# Банк тестовых заданий
QUESTION_BANK = [
    {"id": "Q1", "text": "Вычисление значения дроби со сложением и общим знаменателем", "skills": [1, 4], "N": 4},
    {"id": "Q2", "text": "Нахождение 25% от десятичной дроби 0.8", "skills": [2, 5], "N": 4},
    {"id": "Q3", "text": "Пропорциональное деление и перевод в обыкновенные дроби", "skills": [4, 8], "N": 4},
    {"id": "Q4", "text": "Линейное уравнение с десятичными коэффициентами", "skills": [2, 9], "N": 4},
    {"id": "Q5", "text": "Текстовая задача на движение с составлением уравнения", "skills": [9, 11], "N": 4},
    {"id": "Q6", "text": "Логическая задача на принцип Дирихле", "skills": [7, 10], "N": 4},
    {"id": "Q7", "text": "Расчёт площади составной геометрической фигуры", "skills": [3, 6], "N": 4},
    {"id": "Q8", "text": "Комплексная текстовая задача на совместную работу и проценты", "skills": [5, 8, 11], "N": 4}
]

# ==========================================
# 2. МАТЕМАТИЧЕСКОЕ ЯДРО (МОДЕЛЬ v1.1)
# ==========================================
def get_elimination_params(s, m):
    """Линейная интерполяция параметров элиминации."""
    alpha = s / m if m > 0 else 0.0
    eps = EPS_0 + alpha * (EPS_1 - EPS_0)
    rho = RHO_0 + alpha * (RHO_1 - RHO_0)
    return eps, rho

def entropy_binary(p):
    """Бинарная энтропия Шеннона в битах."""
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

def bayesian_step_2skills(L1, L2, N=4, k=2, is_correct=True, S=0.10, T=0.20):
    """Байесовское обновление для двух навыков с логикой Noisy-AND и G_dyn."""
    G_dyn = 1.0 / (N - k) if (N - k) > 0 else 0.0
    
    p_s0 = (1.0 - L1) * (1.0 - L2)
    p_s1 = L1 * (1.0 - L2) + (1.0 - L1) * L2
    p_s2 = L1 * L2
    
    if is_correct:
        p_c_s0 = G_dyn
        p_c_s1 = G_dyn
        p_c_s2 = 1.0 - S
    else:
        p_c_s0 = 1.0 - G_dyn
        p_c_s1 = 1.0 - G_dyn
        p_c_s2 = S
        
    p_obs = p_s0 * p_c_s0 + p_s1 * p_c_s1 + p_s2 * p_c_s2
    if p_obs == 0:
        return L1, L2
        
    # Маргинализация для L1
    p_L1_1 = L1 * L2 * p_c_s2 + L1 * (1.0 - L2) * p_c_s1
    L1_bayes = p_L1_1 / p_obs
    
    # Маргинализация для L2
    p_L2_1 = L1 * L2 * p_c_s2 + (1.0 - L1) * L2 * p_c_s1
    L2_bayes = p_L2_1 / p_obs
    
    # Марковский шаг обучения (T)
    L1_new = L1_bayes + (1.0 - L1_bayes) * T
    L2_new = L2_bayes + (1.0 - L2_bayes) * T
    
    return float(np.clip(L1_new, 0.01, 0.99)), float(np.clip(L2_new, 0.01, 0.99))

def calculate_eig(question, current_skills):
    """Оценка снижения энтропии для выбора задания."""
    q_skills = question["skills"]
    current_h = sum(entropy_binary(current_skills[i]) for i in q_skills)
    
    if len(q_skills) == 2:
        s1, s2 = q_skills[0], q_skills[1]
        l1, l2 = current_skills[s1], current_skills[s2]
        
        # Симуляция исходов (верно/неверно при k=1)
        l1_c, l2_c = bayesian_step_2skills(l1, l2, k=1, is_correct=True)
        h_c = entropy_binary(l1_c) + entropy_binary(l2_c)
        
        l1_w, l2_w = bayesian_step_2skills(l1, l2, k=1, is_correct=False)
        h_w = entropy_binary(l1_w) + entropy_binary(l2_w)
        
        # Априорная вероятность правильного ответа
        p_corr = (l1 * l2) * (1.0 - SLIP_DEFAULT) + (1.0 - l1 * l2) * (1.0 / 3.0)
        expected_h = p_corr * h_c + (1.0 - p_corr) * h_w
        return max(0.0, current_h - expected_h)
    return 0.15

# ==========================================
# 3. ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ STREAMLIT
# ==========================================
if "skills_p" not in st.session_state:
    st.session_state.skills_p = [0.40] * len(SKILLS)
    # Задаем базовые априорные значения
    st.session_state.skills_p[0] = 0.88  # Арифметика освоена
    st.session_state.skills_p[1] = 0.60  # Общий знаменатель
    st.session_state.skills_p[4] = 0.50  # Обыкновенные дроби

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# 4. ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
# ==========================================
st.title("Графо-байесовская адаптивная система тестирования (BKT v1.1)")
st.caption("Разработка: Марат Ілияс, 10 класс | Областной лицей-интернат для одарённых юношей, г. Экибастуз")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Параметры тестового шага")
    
    selected_q_idx = st.selectbox(
        "Выберите задание из банка:",
        range(len(QUESTION_BANK)),
        format_func=lambda i: f"{QUESTION_BANK[i]['id']}: {QUESTION_BANK[i]['text']}"
    )
    q = QUESTION_BANK[selected_q_idx]
    
    st.info(f"Проверяемые навыки: **{', '.join([SKILLS[i] for i in q['skills']])}**")
    
    k_elim = st.slider("Количество исключенных дистракторов (k):", min_value=0, max_value=3, value=2)
    ans_correct = st.radio("Финальный ответ ученика:", ["Правильный (+)", "Ошибочный (-)"], index=0) == "Правильный (+)"
    
    g_dyn_val = 1.0 / (4 - k_elim) if (4 - k_elim) > 0 else 0.0
    st.metric("Динамическая вероятность угадывания G_dyn", f"{g_dyn_val:.3f}")
    
    if st.button("Выполнить байесовский пересчёт", type="primary"):
        if len(q["skills"]) >= 2:
            s1, s2 = q["skills"][0], q["skills"][1]
            old_l1 = st.session_state.skills_p[s1]
            old_l2 = st.session_state.skills_p[s2]
            
            new_l1, new_l2 = bayesian_step_2skills(
                old_l1, old_l2, N=q["N"], k=k_elim, is_correct=ans_correct
            )
            
            st.session_state.skills_p[s1] = new_l1
            st.session_state.skills_p[s2] = new_l2
            
            st.session_state.history.append({
                "Шаг": len(st.session_state.history) + 1,
                "Задание": q["id"],
                "k": k_elim,
                "Ответ": 1 if ans_correct else 0,
                f"{SKILLS[s1]} (было -> стало)": f"{old_l1:.2f} -> {new_l1:.2f}",
                f"{SKILLS[s2]} (было -> стало)": f"{old_l2:.2f} -> {new_l2:.2f}"
            })
            st.rerun()

    # Оптимизатор EIG
    st.write("---")
    st.subheader("2. Рекомендация следующего задания (EIG Optimizer)")
    eig_scores = []
    for q_item in QUESTION_BANK:
        score = calculate_eig(q_item, st.session_state.skills_p)
        eig_scores.append({"Задание": q_item["id"], "Тема": q_item["text"][:35] + "...", "EIG (бит)": round(score, 4)})
    df_eig = pd.DataFrame(eig_scores).sort_values(by="EIG (бит)", ascending=False)
    st.dataframe(df_eig, hide_index=True, use_container_width=True)

with col_right:
    st.subheader("3. Граф навыков и текущий профиль")
    
    # Построение графа
    fig, ax = plt.subplots(figsize=(8, 6))
    G = nx.DiGraph()
    for i, s in enumerate(SKILLS):
        G.add_node(i, label=s)
    G.add_edges_from(EDGES)
    
    pos = {
        0: (0, 3),
        1: (1, 4), 2: (1, 2), 3: (1, 0),
        4: (2, 4), 5: (2, 2), 6: (2, 0),
        7: (3, 4), 8: (3, 2), 9: (3, 1),
        10: (4, 4), 11: (4, 2)
    }
    
    node_colors = []
    for i in range(len(SKILLS)):
        val = st.session_state.skills_p[i]
        if val >= MASTERY_THRESHOLD:
            node_colors.append("#4CAF50")  # Зеленый - освоен
        elif val >= 0.50:
            node_colors.append("#FFC107")  # Желтый - частичное знание
        else:
            node_colors.append("#F44336")  # Красный - пробел
            
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="#888888", ax=ax)
    
    labels = {i: f"{i}\n{st.session_state.skills_p[i]:.2f}" for i in range(len(SKILLS))}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color="black", ax=ax)
    
    ax.axis("off")
    st.pyplot(fig)
    
    # Таблица навыков
    df_skills = pd.DataFrame({
        "№": range(len(SKILLS)),
        "Навык": SKILLS,
        "P(L)": [round(p, 3) for p in st.session_state.skills_p],
        "Статус": ["Освоен (>=0.85)" if p >= 0.85 else ("В процессе" if p >= 0.50 else "Пробел") for p in st.session_state.skills_p]
    })
    st.dataframe(df_skills, hide_index=True, use_container_width=True, height=220)

if st.session_state.history:
    st.write("---")
    st.subheader("Протокол сессии (Лог тестирования)")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("Сбросить сессию"):
        st.session_state.skills_p = [0.40] * len(SKILLS)
        st.session_state.skills_p[0] = 0.88
        st.session_state.skills_p[1] = 0.60
        st.session_state.skills_p[4] = 0.50
        st.session_state.history = []
        st.rerun()