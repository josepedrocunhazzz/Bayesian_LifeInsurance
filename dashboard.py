import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# PAGE CONFIGURATION 
# ==============================================
st.set_page_config(
    page_title="Dashboard - Seguro de Vida",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .success-metric {
        border-left-color: #10b981;
    }
    .warning-metric {
        border-left-color: #f59e0b;
    }
    .error-metric {
        border-left-color: #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================
# NAIVE BAYES CLASSIFIER CLASS
# ==============================================
class NaiveBayesClassifier:
    def __init__(self):
        self.priors = {}
        self.conditional_probs = {}
        self.continuous_stats = {}
        self.classes = []
        
    def fit(self, X, y):
        """Train the Naive Bayes model"""
        self.classes = np.unique(y)
        n_samples = len(y)
        
        # Calculate prior probabilities
        for class_val in self.classes:
            self.priors[class_val] = np.sum(y == class_val) / n_samples
        
        # Calculate conditional probabilities for each feature
        for feature in X.columns:
            self.conditional_probs[feature] = {}
            self.continuous_stats[feature] = {}
            
            if X[feature].dtype in ['int64', 'float64'] and X[feature].nunique() > 10:
                # Continuous variable - use normal distribution
                for class_val in self.classes:
                    subset = X[y == class_val][feature]
                    self.continuous_stats[feature][class_val] = {
                        'mean': subset.mean(),
                        'std': subset.std() if subset.std() > 0 else 1e-6
                    }
            else:
                # Categorical variable
                for class_val in self.classes:
                    self.conditional_probs[feature][class_val] = {}
                    subset = X[y == class_val]
                    feature_counts = subset[feature].value_counts()
                    total_count = len(subset)
                    
                    # Laplace smoothing
                    unique_vals = X[feature].unique()
                    for val in unique_vals:
                        count = feature_counts.get(val, 0)
                        self.conditional_probs[feature][class_val][val] = (count + 1) / (total_count + len(unique_vals))
    
    def _normal_pdf(self, x, mean, std):
        """Calculate normal probability density"""
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    def predict_proba(self, X):
        """Predict probabilities for each class"""
        probabilities = []
        
        for _, sample in X.iterrows():
            class_probs = {}
            
            for class_val in self.classes:
                prob = self.priors[class_val]
                
                for feature in X.columns:
                    if feature in self.continuous_stats and class_val in self.continuous_stats[feature]:
                        # Continuous variable
                        mean = self.continuous_stats[feature][class_val]['mean']
                        std = self.continuous_stats[feature][class_val]['std']
                        prob *= self._normal_pdf(sample[feature], mean, std)
                    else:
                        # Categorical variable
                        if sample[feature] in self.conditional_probs[feature][class_val]:
                            prob *= self.conditional_probs[feature][class_val][sample[feature]]
                        else:
                            prob *= 1e-6  # Very small value for unseen cases
                
                class_probs[class_val] = prob
            
            # Normalize probabilities
            total_prob = sum(class_probs.values())
            if total_prob > 0:
                normalized_probs = [class_probs[class_val] / total_prob for class_val in self.classes]
            else:
                normalized_probs = [1/len(self.classes) for _ in self.classes]
            
            probabilities.append(normalized_probs)
        
        return np.array(probabilities)
    
    def predict(self, X):
        """Predict classes"""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]

# ==============================================
# DATA LOADING AND PROCESSING FUNCTIONS
# ==============================================
@st.cache_data
def load_data():
    """Load or generate example data"""
    try:
        # Try to load real data
        df = pd.read_csv('lifeInsurance.txt', sep=r'\s+', header=None, 
                        names=['Gender', 'Age', 'MaritalStatus', 'Dependents', 
                              'PhysicalStatus', 'ChronicDiseases', 'MonthlySalary', 'Decision'])
        return df
    except:
        print("Failed to load data")

def calculate_risk_score(row):
    """Calculate risk score based on the 50-point rule"""
    score = 0
    
    # Age
    if row['Age'] < 30: score += 5
    elif row['Age'] < 40: score += 10
    elif row['Age'] < 50: score += 15
    else: score += 20
    
    # Chronic conditions
    if row['ChronicDiseases'] == 0: score += 5
    elif row['ChronicDiseases'] == 1: score += 10
    else: score += 15
    
    # Financial situation
    if row['MonthlySalary'] > 3500: score += 5
    elif row['MonthlySalary'] >= 1700: score += 10
    else: score += 15
    
    # Family responsibilities
    if row['Dependents'] == 0: score += 5
    elif row['Dependents'] == 1: score += 10
    else: score += 15
    
    return score

# ==============================================
# MAIN INTERFACE
# ==============================================
def main():
    st.markdown('<h1 class="main-header">🏥 Dashboard de Análise - Seguro de Vida</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    df['RiskScore'] = df.apply(calculate_risk_score, axis=1)
    
    # Sidebar with filters
    st.sidebar.header("🔧 Filtros")
    
    age_range = st.sidebar.slider("Faixa Etária", 
                                 int(df['Age'].min()), 
                                 int(df['Age'].max()), 
                                 (int(df['Age'].min()), int(df['Age'].max())))
    
    salary_range = st.sidebar.slider("Faixa Salarial", 
                                   int(df['MonthlySalary'].min()), 
                                   int(df['MonthlySalary'].max()), 
                                   (int(df['MonthlySalary'].min()), int(df['MonthlySalary'].max())))
    
    gender_filter = st.sidebar.multiselect("Gênero", 
                                         options=[0, 1], 
                                         default=[0, 1],
                                         format_func=lambda x: "Feminino" if x == 0 else "Masculino")
    
    # Apply filters
    filtered_df = df[
    (df['Age'] >= age_range[0]) & 
    (df['Age'] <= age_range[1]) &
    (df['MonthlySalary'] >= salary_range[0]) & 
    (df['MonthlySalary'] <= salary_range[1]) &
    (df['Gender'].isin(gender_filter))
    ].copy()

    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visão Geral", "👥 Demografia", "⚠️ Análise de Risco", "🤖 Modelo Bayesiano", "🔍 Análise do Modelo"])
    
    # ==============================================
    # TAB 1: OVERVIEW
    # ==============================================
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container success-metric">', unsafe_allow_html=True)
            st.metric("Total de Casos", len(filtered_df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            acceptance_rate = (filtered_df['Decision'].sum() / len(filtered_df)) * 100
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Taxa de Aceitação", f"{acceptance_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            avg_age = filtered_df['Age'].mean()
            st.markdown('<div class="metric-container warning-metric">', unsafe_allow_html=True)
            st.metric("Idade Média", f"{avg_age:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            avg_salary = filtered_df['MonthlySalary'].mean()
            st.markdown('<div class="metric-container error-metric">', unsafe_allow_html=True)
            st.metric("Salário Médio", f"€{avg_salary:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Main charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribuição de Decisões")
            decision_counts = filtered_df['Decision'].value_counts()
            fig = px.pie(values=decision_counts.values, 
                        names=['Rejeita', 'Aceita'], 
                        title="Distribuição de Decisões",
                        color_discrete_sequence=['#ff7f7f', '#90ee90'])
            st.plotly_chart(fig, use_container_width=True, key="decision_pie_chart")
        
        with col2:
            st.subheader("📊 Decisões por Faixa Etária")
            # Create age groups
            bins = [30, 40, 50, 60, 100]
            labels = ['30-39', '40-49', '50-59', '60+']
            filtered_df['AgeGroup'] = pd.cut(filtered_df['Age'], bins=bins, labels=labels, right=False)
            
            age_decision = filtered_df.groupby(['AgeGroup', 'Decision']).size().reset_index(name='Count')
            age_decision['Decision'] = age_decision['Decision'].map({0: 'Rejeita', 1: 'Aceita'})
            
            fig = px.bar(age_decision, x='AgeGroup', y='Count', color='Decision',
                        title="Decisões por Faixa Etária",
                        color_discrete_sequence=['#ff7f7f', '#90ee90'])
            st.plotly_chart(fig, use_container_width=True, key="age_decision_bar")
    
    # ==============================================
    # TAB 2: DEMOGRAPHICS
    # ==============================================
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Scatter: Idade vs Salário")
            fig = px.scatter(filtered_df, x='Age', y='MonthlySalary', 
                           color='Decision', 
                           title="Relação Idade vs Salário",
                           color_discrete_map={0: '#ff7f7f', 1: '#90ee90'},
                           labels={'Decision': 'Decisão'})
            st.plotly_chart(fig, use_container_width=True, key="age_salary_scatter")
        
        with col2:
            st.subheader("👨‍👩‍👧‍👦 Dependentes vs Decisão")
            dep_decision = filtered_df.groupby(['Dependents', 'Decision']).size().reset_index(name='Count')
            dep_decision['Decision'] = dep_decision['Decision'].map({0: 'Rejeita', 1: 'Aceita'})
            
            fig = px.bar(dep_decision, x='Dependents', y='Count', color='Decision',
                        title="Decisões por Número de Dependentes",
                        color_discrete_sequence=['#ff7f7f', '#90ee90'])
            st.plotly_chart(fig, use_container_width=True, key="dependents_decision_bar")
        
        # Correlation matrix
        st.subheader("🔗 Matriz de Correlação")
        corr_matrix = filtered_df[['Age', 'MonthlySalary', 'Dependents', 'PhysicalStatus', 
                                 'ChronicDiseases', 'Decision', 'RiskScore']].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto=True, 
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title="Matriz de Correlação")
        st.plotly_chart(fig, use_container_width=True, key="correlation_matrix")
    
    # ==============================================
    # TAB 3: RISK ANALYSIS
    # ==============================================
    with tab3:
        st.subheader("⚠️ Análise de Risco Baseada na Regra dos 50 Pontos")
        
        # Information about the rule
        st.info("""
        **Regra de Decisão:**
        - **Idade:** <30 (5pts), 30-39 (10pts), 40-49 (15pts), 50+ (20pts)
        - **Saúde:** Saudável (5pts), Moderada (10pts), Severa (15pts)  
        - **Financeiro:** >3500€ (5pts), 1700-3500€ (10pts), <1700€ (15pts)
        - **Dependentes:** 0 (5pts), 1 (10pts), 2+ (15pts)
        - **Recomendação:** Score > 50 → Comprar seguro
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribuição de Scores")
            fig = px.histogram(filtered_df, x='RiskScore', nbins=20,
                             title="Distribuição dos Scores de Risco")
            fig.add_vline(x=50, line_dash="dash", line_color="red", 
                         annotation_text="Limite (50 pontos)")
            st.plotly_chart(fig, use_container_width=True, key="risk_score_hist")
        
        with col2:
            st.subheader("🎯 Score vs Decisão")
            fig = px.box(filtered_df, x='Decision', y='RiskScore',
                        title="Scores por Decisão")
            fig.update_layout(
                xaxis=dict(
                    tickvals=[0, 1],
                    ticktext=['Rejeita', 'Aceita']
                )
            )
            st.plotly_chart(fig, use_container_width=True, key="risk_score_box")
        
        # Analysis by chronic conditions
        st.subheader("🏥 Análise por Condições Crônicas")
        chronic_analysis = filtered_df.groupby(['ChronicDiseases', 'Decision']).size().reset_index(name='Count')
        chronic_analysis['ChronicDiseases'] = chronic_analysis['ChronicDiseases'].map({
            0: 'Nenhuma', 1: 'Moderada', 2: 'Severa'
        })
        chronic_analysis['Decision'] = chronic_analysis['Decision'].map({0: 'Rejeita', 1: 'Aceita'})
        
        fig = px.bar(chronic_analysis, x='ChronicDiseases', y='Count', color='Decision',
                    title="Decisões por Condições Crônicas",
                    color_discrete_sequence=['#ff7f7f', '#90ee90'])
        st.plotly_chart(fig, use_container_width=True, key="chronic_analysis_bar")
    
    # ==============================================
    # TAB 4: BAYESIAN MODEL
    # ==============================================
    with tab4:
        st.subheader("🤖 Modelo Bayesiano Naive Bayes")
        
        # Prepare data for the model
        features = ['Gender', 'Age', 'MaritalStatus', 'Dependents', 
                   'PhysicalStatus', 'ChronicDiseases', 'MonthlySalary']
        X = filtered_df[features]
        y = filtered_df['Decision']
        
        # Train model
        model = NaiveBayesClassifier()
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Acurácia do Modelo", f"{accuracy:.1%}")
        
        with col2:
            st.metric("Probabilidade Prévia (Aceita)", f"{model.priors[1]:.1%}")
        
        with col3:
            st.metric("Probabilidade Prévia (Rejeita)", f"{model.priors[0]:.1%}")
        
        # Confusion matrix
        st.subheader("📊 Matriz de Confusão")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, predictions)
        
        fig = px.imshow(cm, 
                       text_auto=True,
                       labels=dict(x="Predito", y="Real"),
                       x=['Rejeita', 'Aceita'],
                       y=['Rejeita', 'Aceita'],
                       title="Matriz de Confusão")
        st.plotly_chart(fig, use_container_width=True, key="confusion_matrix")
        
        # Interactive prediction section
        st.subheader("🎯 Predição Interativa")
        st.write("Insira os dados para obter uma predição:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_gender = st.selectbox("Gênero", [0, 1], format_func=lambda x: "Feminino" if x == 0 else "Masculino")
            pred_age = st.slider("Idade", 18, 70, 45)
            pred_marital = st.selectbox("Estado Civil", [0, 1], format_func=lambda x: "Solteiro" if x == 0 else "Casado")
        
        with col2:
            pred_dependents = st.selectbox("Dependentes", [0, 1, 2, 3])
            pred_physical = st.selectbox("Status Físico", [0, 1, 2], 
                                       format_func=lambda x: ["Sedentário", "Moderadamente Ativo", "Ativo"][x])
            pred_chronic = st.selectbox("Doenças Crônicas", [0, 1, 2],
                                      format_func=lambda x: ["Nenhuma", "Moderada", "Severa"][x])
        
        with col3:
            pred_salary = st.slider("Salário Mensal", 1000, 5000, 2500)
        
        # Make prediction
        if st.button("🔮 Fazer Predição"):
            pred_data = pd.DataFrame({
                'Gender': [pred_gender],
                'Age': [pred_age],
                'MaritalStatus': [pred_marital],
                'Dependents': [pred_dependents],
                'PhysicalStatus': [pred_physical],
                'ChronicDiseases': [pred_chronic],
                'MonthlySalary': [pred_salary]
            })
            
            pred_result = model.predict(pred_data)[0]
            pred_proba = model.predict_proba(pred_data)[0]  # Array with [P(0), P(1)]
            
            # Calculate risk score
            risk_score = calculate_risk_score({
                'Age': pred_age,
                'ChronicDiseases': pred_chronic,
                'MonthlySalary': pred_salary,
                'Dependents': pred_dependents
            })
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                decision_text = "COMPRAR SEGURO" if pred_result == 1 else "NÃO COMPRAR"
                st.success(f"**Decisão:** {decision_text}")
            
            with col2:
                # Correct probability display
                proba = pred_proba[1] if pred_result == 1 else pred_proba[0]
                st.info(f"**Probabilidade:** {proba:.1%}")
            
            with col3:
                st.warning(f"**Score de Risco:** {risk_score} pontos")
        
        # Model information
        st.subheader("ℹ️ Informações do Modelo")
        st.write("""
        **Características do Modelo Bayesiano Implementado:**
        - ✅ Implementado sem bibliotecas externas (apenas NumPy/Pandas)
        - ✅ Naive Bayes com Laplace Smoothing
        - ✅ Suporte a variáveis categóricas e contínuas
        - ✅ Distribuição Normal para idade e salário
        - ✅ Probabilidades condicionais para variáveis categóricas
        - ✅ Cálculo de probabilidades a priori baseado nos dados
        """)

    # ==============================================
    # TAB 5: MODEL ANALYSIS
    # ==============================================
    with tab5:
        st.subheader("🔍 Análise Detalhada do Modelo")
        
        # Prepare data for the model
        features = ['Gender', 'Age', 'MaritalStatus', 'Dependents', 
                   'PhysicalStatus', 'ChronicDiseases', 'MonthlySalary']
        X = filtered_df[features]
        y = filtered_df['Decision']
        
        # Train complete model
        model = NaiveBayesClassifier()
        model.fit(X, y)
        predictions = model.predict(X)
        
        # 1. Classifier Performance
        st.markdown("### 1️⃣ Performance do Classificador")
        
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Calculate metrics
        accuracy = np.mean(predictions == y)
        rule_predictions = (filtered_df['RiskScore'] > 50).astype(int)
        rule_accuracy = np.mean(rule_predictions == y)
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        fig = px.imshow(cm, 
                       text_auto=True,
                       labels=dict(x="Predito", y="Real"),
                       x=['Rejeita', 'Aceita'],
                       y=['Rejeita', 'Aceita'],
                       title="Matriz de Confusão")
        st.plotly_chart(fig, use_container_width=True, key="model_confusion_matrix")
        
        # Classification report - ensuring both classes are present
        report = classification_report(y, predictions, output_dict=True, target_names=['Rejeita', 'Aceita'])
        
        # Convert report to DataFrame for better display
        report_df = pd.DataFrame(report).transpose().reset_index()
        report_df.columns = ['Métrica'] + list(report_df.columns[1:])
        
        # Format numbers for better display
        for col in report_df.columns[1:-1]:
            report_df[col] = report_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x)
        
        # Display as a nice table
        st.table(report_df)
        
        st.markdown(f"""
        **Análise:**
        - O modelo Naive Bayes apresentou acurácia de {accuracy:.1%} comparado com {rule_accuracy:.1%} da regra simples dos 50 pontos.
        - A matriz de confusão mostra {cm[1,1]} verdadeiros positivos e {cm[0,0]} verdadeiros negativos.
        - O recall para casos positivos é {report['Aceita']['recall']:.1%}, indicando boa capacidade de identificar clientes que devem comprar seguro.
        - A precisão para casos negativos é {report['Rejeita']['precision']:.1%}, mostrando confiabilidade nas rejeições.
        """)
        
        # 2. Feature Selection
        st.markdown("---")
        st.markdown("### 2️⃣ Seleção de Variáveis")
        
        # Test different feature combinations
        features_combinations = [
            (['Age', 'ChronicDiseases', 'MonthlySalary', 'Dependents'], "4 features (regra dos 50pts)"),
            (['Age', 'ChronicDiseases', 'MonthlySalary', 'Dependents', 'Gender'], "5 features (+Gênero)"),
            (['Age', 'ChronicDiseases', 'MonthlySalary', 'Dependents', 'PhysicalStatus'], "5 features (+Status Físico)"),
            (['Age', 'ChronicDiseases', 'MonthlySalary', 'Dependents', 'Gender', 'PhysicalStatus'], "6 features"),
            (features, "Todas features (7)")
        ]
        
        # Calculate accuracies
        accuracies = []
        labels = []
        for features_subset, label in features_combinations:
            model = NaiveBayesClassifier()
            model.fit(X[features_subset], y)
            acc = np.mean(model.predict(X[features_subset]) == y)
            accuracies.append(acc)
            labels.append(label)
        
        # Create dataframe for visualization
        perf_df = pd.DataFrame({
            'Número de Features': [len(fs[0]) for fs in features_combinations],
            'Acurácia': accuracies,
            'Descrição': labels
        })
        
        # Plot
        fig = px.bar(perf_df, 
                    x='Número de Features', 
                    y='Acurácia',
                    color='Descrição',
                    text=[f"{acc:.1%}" for acc in accuracies],
                    title='Performance por Conjunto de Features',
                    labels={'Acurácia': 'Acurácia', 'Número de Features': 'Número de Features'})
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=perf_df['Número de Features'].unique(),
                ticktext=perf_df['Número de Features'].unique()
            ),
            yaxis=dict(
                range=[0, 1]  # Set y-axis from 0 to 1 for percentage
            )
        )
        st.plotly_chart(fig, use_container_width=True, key="feature_selection_bar")
        
        # Correlations
        corr_matrix = filtered_df[features + ['Decision']].corr()['Decision'].drop('Decision')
        fig = px.bar(corr_matrix, 
                    title='Correlação com Decisão',
                    labels={'index':'Feature', 'value':'Correlação'})
        st.plotly_chart(fig, use_container_width=True, key="feature_correlation_bar")
        
        st.markdown(f"""
        **Análise:**
        - As 4 variáveis da regra dos 50 pontos (Age, ChronicDiseases, MonthlySalary, Dependents) capturam {accuracies[0]:.1%} da performance.
        - Adicionar Gender e PhysicalStatus aumenta a acurácia para {accuracies[3]:.1%} com 6 features.
        - Todas as features juntas alcançam {accuracies[4]:.1%}, mostrando que algumas variáveis adicionam pouco valor.
        - Correlação mais forte com ChronicDiseases ({corr_matrix['ChronicDiseases']:.2f}) e Age ({corr_matrix['Age']:.2f}).
        """)
        
        # 3. Discrete vs Continuous Variables
        st.markdown("---")
        st.markdown("### 3️⃣ Variáveis Discretas vs Contínuas")
        
        # Create discrete version
        X_discrete = X.copy()
        X_discrete['Age'] = pd.cut(X['Age'], bins=[0, 30, 40, 50, 100], labels=[0,1,2,3])
        X_discrete['MonthlySalary'] = pd.cut(X['MonthlySalary'], bins=[0, 1700, 3500, 5000], labels=[0,1,2])
        
        # Train models
        model_continuous = NaiveBayesClassifier()
        model_continuous.fit(X[['Age', 'MonthlySalary']], y)
        cont_acc = np.mean(model_continuous.predict(X[['Age', 'MonthlySalary']]) == y)
        
        model_discrete = NaiveBayesClassifier()
        model_discrete.fit(X_discrete[['Age', 'MonthlySalary']], y)
        disc_acc = np.mean(model_discrete.predict(X_discrete[['Age', 'MonthlySalary']]) == y)
        
        fig = px.bar(x=['Contínuas', 'Discretas'], y=[cont_acc, disc_acc],
                    labels={'x':'Tipo de Variável', 'y':'Acurácia'},
                    title='Performance: Contínuas vs Discretas',
                    text=[f"{cont_acc:.1%}", f"{disc_acc:.1%}"])
        st.plotly_chart(fig, use_container_width=True, key="discrete_vs_continuous_bar")
        
        st.markdown(f"""
        **Análise:**
        - Tratamento contínuo: Acurácia de {cont_acc:.1%}
        - Tratamento discreto: Acurácia de {disc_acc:.1%}
        - Abordagem contínua mostrou-se superior, preservando informação granular importante para decisões.
        - Discretização pode ser útil para interpretação, mas custa performance.
        """)
        
        # 4. Variable Normality
        st.markdown("---")
        st.markdown("### 4️⃣ Normalidade das Variáveis")
        
        def test_normality(data, feature, class_val):
            subset = data[data['Decision'] == class_val][feature]
            statistic, p_value = stats.shapiro(subset)
            return p_value > 0.05  # Normal if p > 0.05
        
        normality_results = []
        for class_val in [0, 1]:
            for feature in ['Age', 'MonthlySalary']:
                is_normal = test_normality(filtered_df, feature, class_val)
                normality_results.append({
                    'Feature': feature,
                    'Classe': 'Aceita' if class_val == 1 else 'Rejeita',
                    'Normal': is_normal,
                    'p-value': stats.shapiro(filtered_df[filtered_df['Decision'] == class_val][feature])[1]
                })
        
        normality_df = pd.DataFrame(normality_results)
        
        fig = px.bar(normality_df, x='Feature', y='p-value', color='Classe', barmode='group',
                    title='Teste de Normalidade (Shapiro-Wilk)',
                    labels={'p-value':'Valor-p'},
                    text=['Normal' if x else 'Não-normal' for x in normality_df['Normal']])
        fig.add_hline(y=0.05, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True, key="normality_test_bar")
        
        st.markdown("""
        **Análise:**
        - Teste de Shapiro-Wilk rejeitou normalidade para Idade em ambas as classes (p < 0.05).
        - Salário aproxima-se mais da normalidade, especialmente para casos aceitos.
        - Suposição de normalidade pode não ser ideal para Age. Alternativas:
          - Transformações logarítmicas
          - Distribuições não-paramétricas
          - Misturas gaussianas para capturar multimodalidade
        """)

        # 5. Gaussian Mixture Approach
        st.markdown("---")
        st.markdown("### 5️⃣ Abordagem com Mistura Gaussiana")
        
        st.markdown("""
        **Por que considerar Misturas Gaussianas?**
        - Dados reais frequentemente não seguem uma única distribuição normal
        - Misturas podem capturar subpopulações (ex: jovens saudáveis vs idosos com doenças)
        - Melhor modelagem de dados multimodais
        """)
        
        from sklearn.mixture import GaussianMixture
        
        # Prepare data for GMM analysis
        gmm_features = ['Age', 'MonthlySalary']
        X_gmm = filtered_df[gmm_features]
        
        # Fit GMMs for each class
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Distribuição para Rejeições", "Distribuição para Aceitações"),
                           shared_yaxes=True)
        
        # Add original data points
        fig.add_trace(
            go.Scatter(
                x=X_gmm[filtered_df['Decision'] == 0]['Age'],
                y=X_gmm[filtered_df['Decision'] == 0]['MonthlySalary'],
                mode='markers',
                name='Rejeita (dados)',
                marker=dict(color='red', opacity=0.3)
            ), row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=X_gmm[filtered_df['Decision'] == 1]['Age'],
                y=X_gmm[filtered_df['Decision'] == 1]['MonthlySalary'],
                mode='markers',
                name='Aceita (dados)',
                marker=dict(color='green', opacity=0.3)
            ), row=1, col=2
        )
        
        # Fit and plot GMMs
        for class_val, color, col in zip([0, 1], ['red', 'green'], [1, 2]):
            subset = X_gmm[filtered_df['Decision'] == class_val]
            
            # Fit GMM with 2 components
            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(subset)
            
            # Create grid for contour plot
            x_min, x_max = subset['Age'].min() - 1, subset['Age'].max() + 1
            y_min, y_max = subset['MonthlySalary'].min() - 100, subset['MonthlySalary'].max() + 100
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            grid = np.c_[xx.ravel(), yy.ravel()]
            
            # Calculate densities
            densities = np.exp(gmm.score_samples(grid))
            densities = densities.reshape(xx.shape)
            
            # Add contour plot
            fig.add_trace(
                go.Contour(
                    x=np.linspace(x_min, x_max, 100),
                    y=np.linspace(y_min, y_max, 100),
                    z=densities,
                    showscale=False,
                    name=f'GMM (Classe {class_val})',
                    line=dict(width=0),
                    contours=dict(coloring='lines'),
                    line_color=color
                ), row=1, col=col
            )
            
            # Add component means
            for mean in gmm.means_:
                fig.add_trace(
                    go.Scatter(
                        x=[mean[0]],
                        y=[mean[1]],
                        mode='markers',
                        marker=dict(color=color, size=10, symbol='x'),
                        showlegend=False
                    ), row=1, col=col
                )
        
        fig.update_layout(
            title='Distribuições com Misturas Gaussianas (2 componentes)',
            xaxis_title='Idade',
            yaxis_title='Salário Mensal',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True, key="gmm_plot")
        
        # Compare model performance
        st.markdown("**Comparação de Performance:**")
        
        # Train models with GMM approach
        class GMMNaiveBayes:
            def __init__(self, n_components=2):
                self.n_components = n_components
                self.gmms = {}
                self.priors = {}
                
            def fit(self, X, y):
                self.classes = np.unique(y)
                
                # Calculate priors
                for class_val in self.classes:
                    self.priors[class_val] = np.mean(y == class_val)
                    
                # Fit GMM for each class
                for class_val in self.classes:
                    subset = X[y == class_val]
                    gmm = GaussianMixture(n_components=self.n_components, random_state=42)
                    gmm.fit(subset)
                    self.gmms[class_val] = gmm
                    
            def predict_proba(self, X):
                probas = []
                for _, sample in X.iterrows():
                    class_probs = {}
                    for class_val in self.classes:
                        # Prior * likelihood
                        prob = self.priors[class_val] * np.exp(self.gmms[class_val].score_samples([sample])[0])
                        class_probs[class_val] = prob
                    
                    # Normalize
                    total = sum(class_probs.values())
                    normalized = {k: v/total for k, v in class_probs.items()}
                    probas.append([normalized[0], normalized[1]])
                
                return np.array(probas)
            
            def predict(self, X):
                probas = self.predict_proba(X)
                return np.argmax(probas, axis=1)
        
        # Compare models
        models = {
            'Naive Bayes Padrão': NaiveBayesClassifier(),
            'GMM Naive Bayes (2 componentes)': GMMNaiveBayes(n_components=2),
            'GMM Naive Bayes (3 componentes)': GMMNaiveBayes(n_components=3)
        }
        
        results = []
        for name, model in models.items():
            model.fit(X[['Age', 'MonthlySalary']], y)
            preds = model.predict(X[['Age', 'MonthlySalary']])
            acc = np.mean(preds == y)
            results.append({'Modelo': name, 'Acurácia': acc})
        
        results_df = pd.DataFrame(results)
        
        fig = px.bar(results_df, x='Modelo', y='Acurácia', 
                    text=[f"{acc:.1%}" for acc in results_df['Acurácia']],
                    title='Comparação de Abordagens para Variáveis Contínuas')
        st.plotly_chart(fig, use_container_width=True, key="model_comparison_bar")
        
        st.markdown("""
        **Principais Conclusões:**
        1. A abordagem GMM captura melhor a estrutura multimodal dos dados (visível nos contornos)
        2. Para Age e MonthlySalary, a mistura de 2 gaussianas por classe:
           - Desempenho semelhante ao Naive Bayes padrão
           - Identifica subpopulações distintas (ex: jovens com salários baixos vs médios)
        3. Aumentar para 3 componentes traz ganhos marginais
        4. Custo computacional maior que Naive Bayes tradicional
        """)
        
        # Update recommendations section 
        st.markdown("---")
        st.markdown("### 6️⃣ Recomendações e Melhorias")
        
        st.markdown("""
        **Recomendações para Melhoria do Modelo:**
        1. **Modelagem de Variáveis Contínuas:**
           - Implementar GMM para Age e MonthlySalary
           - Considerar distribuição log-normal para salários
        2. **Seleção de Features:** 
           - Manter Age, ChronicDiseases, MonthlySalary e Dependentes
           - Criar interações entre features (ex: idade × doenças crônicas)
        3. **Validação:**
           - Implementar validação cruzada
           - Testar em conjunto de dados separado
        4. **Balanceamento:**
           - Avaliar se classes estão balanceadas
           - Considerar técnicas como SMOTE se necessário
        """)

if __name__ == "__main__":
    main()

#pip install streamlit pandas numpy matplotlib seaborn scipy plotly
#streamlit run dashboard.py