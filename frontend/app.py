import streamlit as st
import requests

st.title("Система кредитного скоринга")

# Создаем форму для ввода данных
with st.form("credit_form"):
    st.header("Основные параметры заявки")
    col1, col2 = st.columns(2)
    with col1:
        loan_amnt = st.number_input("Сумма кредита ($)", min_value=0)
        int_rate = st.number_input("Процентная ставка (%)", min_value=0.0)
        term = st.selectbox("Срок кредита (мес)", [36, 60])
        annual_inc = st.number_input("Годовой доход ($)", min_value=0)
        emp_length = st.number_input("Стаж работы (лет)", min_value=0)
        fico_range_low = st.number_input("Кредитный скоринг FICO", min_value=300, max_value=850)

    with col2:
        dti = st.number_input("Отношение долга к доходу (%)", min_value=0.0)
        open_acc = st.number_input("Открытые кредитные линии", min_value=0)
        delinq_2yrs = st.number_input("Просрочки за 2 года", min_value=0)
        total_acc = st.number_input("Всего кредитных счетов", min_value=0)
        revol_util = st.number_input("Использование кредитного лимита (%)", min_value=0.0, max_value=100.0)
        inq_last_6mths = st.number_input("Кредитные запросы (последние 6 мес)", min_value=0)

    st.header("Дополнительная информация")
    col3, col4 = st.columns(2)
    with col3:
        mort_acc = st.number_input("Ипотечные счета", min_value=0)
        pub_rec_bankruptcies = st.number_input("Банкротства в истории", min_value=0)
        Source_Verified = st.checkbox("Доход подтвержден источниками")
        Verified = st.checkbox("Доход полностью верифицирован")

    with col4:
        OWN = st.checkbox("Владеет жильем")
        RENT = st.checkbox("Арендует жилье")
        OTHER = st.checkbox("Другая ситуация с жильем")
        purpose_home_improvement = st.checkbox("Цель: ремонт жилья, машины")
        purpose_debt_consolidation = st.checkbox("Цель: объединение долгов")
        purpose_other = st.checkbox("Другая цель кредита")

    submitted = st.form_submit_button("Проверить одобрение")

if submitted:
    # Подготовка данных для отправки
    data = {
        "loan_amnt": float(loan_amnt),
        "int_rate": float(int_rate),
        "term": int(term),
        "annual_inc": float(annual_inc),
        "Source_Verified": bool(Source_Verified),
        "Verified": bool(Verified),
        "emp_length": float(emp_length),
        "dti": float(dti),
        "fico_range_low": float(fico_range_low),
        "open_acc": float(open_acc),
        "delinq_2yrs": float(delinq_2yrs),
        "mort_acc": float(mort_acc),
        "total_acc": float(total_acc),
        "revol_util": float(revol_util),
        "inq_last_6mths": float(inq_last_6mths),
        "OWN": bool(OWN),
        "RENT": bool(RENT),
        "OTHER": bool(OTHER),
        "purpose_home_improvement": bool(purpose_home_improvement),
        "purpose_debt_consolidation": bool(purpose_debt_consolidation),
        "purpose_other": bool(purpose_other),
        "pub_rec_bankruptcies": float(pub_rec_bankruptcies)
    }

    # Отправка запроса к API
    response = requests.post("http://backend:8000/predict/", json=data)

    if response.status_code == 200:
        result = response.json()
        if "decision" in result:
            # Если коедит одобрен - зеленый цвет уведомления, нет - красный
            if "не одобрен" in result['decision'].lower():
                st.error(f"Результат: {result['decision']}")  # Красный
            else:
                st.success(f"Результат: {result['decision']}")  # Зеленый
            st.info(f"Вероятность одобрения: {result['probability']:.2%}")
            st.info(f"Пороговое значение: {result['threshold']}")
        else:
            st.error("Ошибка в ответе сервера")
    else:
        st.error(f"Ошибка при запросе: {response.text}")