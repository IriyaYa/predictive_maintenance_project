import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    st.title("Прогнозирование отказов оборудования")

    # Инициализация состояния
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.is_trained = False
        st.session_state.scaler = StandardScaler()
        st.session_state.data_loaded = False
        st.session_state.expected_columns = []

    # Загрузка данных - только один раз
    if not st.session_state.data_loaded:
        data = None

        # Попытка загрузки из CSV
        try:
            data = pd.read_csv("data/predictive_maintenance.csv")
            st.success("Данные загружены из локального файла")
        except:
            st.warning("Локальный файл не найден. Пытаюсь загрузить с UCI...")

            # Загрузка с UCI Repository
            try:
                from ucimlrepo import fetch_ucirepo
                dataset = fetch_ucirepo(id=601)

                # Создаем DataFrame только с нужными столбцами
                data = pd.DataFrame({
                    'Type': dataset.data.features['Type'],
                    'Air temperature [K]': dataset.data.features['Air temperature [K]'],
                    'Process temperature [K]': dataset.data.features['Process temperature [K]'],
                    'Rotational speed [rpm]': dataset.data.features['Rotational speed [rpm]'],
                    'Torque [Nm]': dataset.data.features['Torque [Nm]'],
                    'Tool wear [min]': dataset.data.features['Tool wear [min]'],
                    'Machine failure': dataset.data.targets['Machine failure']
                })
                st.success("Данные успешно загружены с UCI Repository!")
            except Exception as e:
                st.error(f"Ошибка загрузки данных: {str(e)}")
                st.stop()

        if data is None:
            st.error("Не удалось загрузить данные")
            st.stop()

        # Удаляем все дополнительные столбцы (TWF, HDF и др.)
        required_columns = [
            'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure'
        ]

        columns_to_remove = [col for col in data.columns if col not in required_columns]
        if columns_to_remove:
            st.warning(f"Удалены дополнительные столбцы: {', '.join(columns_to_remove)}")
            data = data.drop(columns=columns_to_remove)

        # Сохраняем данные в session_state
        st.session_state.data = data
        st.session_state.data_loaded = True

    # Используем сохраненные данные
    data = st.session_state.data

    # Проверка наличия обязательных столбцов
    required_columns = [
        'Type', 'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure'
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Отсутствуют важные столбцы: {', '.join(missing_columns)}")
        st.write("Попробуйте перезагрузить данные или использовать локальный файл")
        st.stop()

    # Показать структуру данных
    if st.checkbox("Показать информацию о данных"):
        st.subheader("Структура данных")
        st.write(f"Количество строк: {data.shape[0]}, столбцов: {data.shape[1]}")
        st.write("Первые 3 строки:")
        st.write(data.head(3))
        st.write("Типы данных:")
        st.write(data.dtypes)

    # Предобработка данных
    st.subheader("Предобработка данных")

    # Преобразование категориальной переменной Type
    if 'Type' in data.columns:
        # Проверяем уникальные значения
        unique_types = data['Type'].unique()
        st.info(f"Уникальные значения в столбце 'Type': {unique_types}")

        # Создаем маппинг в зависимости от типа данных
        if data['Type'].dtype == 'object':
            # Если данные текстовые (L, M, H)
            type_mapping = {'L': 0, 'M': 1, 'H': 2}
            data['Type'] = data['Type'].map(type_mapping)
            st.success("Столбец 'Type' преобразован в числовой формат")
        else:
            # Если данные уже числовые
            st.warning("Столбец 'Type' уже содержит числовые значения. Преобразование не требуется.")

            # Проверяем диапазон значений и при необходимости нормализуем
            if data['Type'].min() < 0 or data['Type'].max() > 2:
                st.warning("Значения выходят за ожидаемый диапазон (0-2). Выполняю нормализацию...")
                data['Type'] = data['Type'] % 3  # Приводим к диапазону 0-2
                st.success("Значения нормализованы к диапазону 0-2")
    else:
        st.error("Столбец 'Type' не найден в данных!")
        st.stop()

    # Масштабирование числовых признаков
    numerical_cols = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]

    # Проверяем, что все числовые столбцы присутствуют
    missing_numerical = [col for col in numerical_cols if col not in data.columns]
    if missing_numerical:
        st.error(f"Отсутствуют числовые столбцы: {', '.join(missing_numerical)}")
        st.stop()

    # Применяем масштабирование
    try:
        st.session_state.scaler.fit(data[numerical_cols])
        data[numerical_cols] = st.session_state.scaler.transform(data[numerical_cols])
        st.success("Числовые признаки успешно масштабированы")
    except Exception as e:
        st.error(f"Ошибка при масштабировании: {str(e)}")
        st.stop()

    # Разделение данных
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Сохраняем названия столбцов для прогноза
    st.session_state.expected_columns = X_train.columns.tolist()

    # Обучение модели
    st.subheader("Обучение модели")

    if st.button("Обучить модель Random Forest"):
        with st.spinner("Модель обучается..."):
            try:
                # Создание и обучение модели
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight='balanced'
                )
                model.fit(X_train, y_train)

                # Сохранение модели
                st.session_state.model = model
                st.session_state.is_trained = True

                # Оценка модели
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Вывод результатов
                st.success(f"Модель успешно обучена! Точность: {accuracy:.2%}")

                # Матрица ошибок
                st.subheader("Матрица ошибок")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['Нет отказа', 'Отказ'],
                            yticklabels=['Нет отказа', 'Отказ'])
                ax.set_xlabel('Предсказание')
                ax.set_ylabel('Факт')
                ax.set_title('Матрица ошибок')
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Ошибка при обучении модели: {str(e)}")

    # Разделитель
    st.markdown("---")

    # Прогнозирование для нового оборудования
    st.header("Прогнозирование для нового оборудования")

    with st.form("prediction_form"):
        st.info("Заполните параметры оборудования для прогноза")

        col1, col2 = st.columns(2)

        with col1:
            equipment_type = st.selectbox(
                "Тип оборудования",
                ["L (Низкое качество)", "M (Среднее качество)", "H (Высокое качество)"]
            )
            air_temp = st.number_input(
                "Температура воздуха (K)",
                min_value=250.0,
                max_value=350.0,
                value=300.0
            )
            process_temp = st.number_input(
                "Температура процесса (K)",
                min_value=250.0,
                max_value=350.0,
                value=310.0
            )

        with col2:
            rotation_speed = st.number_input(
                "Скорость вращения (об/мин)",
                min_value=1000,
                max_value=3000,
                value=1500
            )
            torque = st.number_input(
                "Крутящий момент (Нм)",
                min_value=0.0,
                max_value=100.0,
                value=40.0
            )
            tool_wear = st.number_input(
                "Износ инструмента (мин)",
                min_value=0,
                max_value=300,
                value=50
            )

        submit_button = st.form_submit_button("Предсказать отказ")

        if submit_button:
            if not st.session_state.get('is_trained', False):
                st.error("❌ Сначала обучите модель, нажав кнопку 'Обучить модель' выше!")
            else:
                try:
                    # Преобразование введенных данных
                    type_map = {
                        "L (Низкое качество)": 0,
                        "M (Среднее качество)": 1,
                        "H (Высокое качество)": 2
                    }

                    # Создаем DataFrame с введенными данными
                    input_data = pd.DataFrame({
                        'Type': [type_map[equipment_type]],
                        'Air temperature [K]': [air_temp],
                        'Process temperature [K]': [process_temp],
                        'Rotational speed [rpm]': [rotation_speed],
                        'Torque [Nm]': [torque],
                        'Tool wear [min]': [tool_wear]
                    })

                    # Добавляем недостающие столбцы с нулевыми значениями
                    for col in st.session_state.expected_columns:
                        if col not in input_data.columns:
                            input_data[col] = 0.0

                    # Упорядочиваем столбцы как при обучении
                    input_data = input_data[st.session_state.expected_columns]

                    # Масштабируем числовые признаки
                    numerical_cols = [
                        'Air temperature [K]', 'Process temperature [K]',
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
                    ]

                    input_data_scaled = input_data.copy()
                    input_data_scaled[numerical_cols] = st.session_state.scaler.transform(
                        input_data[numerical_cols]
                    )

                    # Прогноз
                    model = st.session_state.model
                    prediction = model.predict(input_data_scaled)[0]
                    probability = model.predict_proba(input_data_scaled)[0][1]

                    # Результат
                    st.subheader("Результат прогнозирования")
                    if prediction == 1:
                        st.error(f"⚠️ **Вероятность отказа оборудования: {probability:.1%}**")
                        st.markdown("Рекомендуется провести техническое обслуживание!")
                    else:
                        st.success(f"✅ **Вероятность отказа оборудования: {probability:.1%}**")
                        st.markdown("Оборудование работает в штатном режиме")

                    # Дополнительная информация
                    st.markdown("---")
                    st.info("Интерпретация вероятности отказа:")
                    st.progress(float(probability))

                    if probability < 0.3:
                        st.write("Низкий риск - плановое обслуживание не требуется")
                    elif probability < 0.7:
                        st.write("Средний риск - рекомендуется проверить оборудование")
                    else:
                        st.write("Высокий риск - требуется срочное обслуживание")

                except Exception as e:
                    st.error(f"Ошибка при прогнозировании: {str(e)}")
                    st.write("Ожидаемые столбцы:", st.session_state.expected_columns)
                    st.write("Фактические столбцы:", list(input_data.columns))


if __name__ == "__main__":
    main()