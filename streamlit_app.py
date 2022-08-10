import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import requests
from PIL import Image
import streamlit.components.v1 as components

# Load data
df = pd.read_csv('X_train_features.csv')

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Le menu principal",
        options=["Prédiction des clients", "Prédiction des nouvaux clients", 'Comparaison'],
        icons=["house", "person", "people fill"],
        menu_icon="cast",
        default_index=0)
    image = Image.open(r'logo_entreprise.png')

    st.image(image, caption='Prêt à depenser')

# Home page
if selected == "Prédiction des clients":

    html_temp ="""
       <div style="background-color:tomato;padding:10px">
       <h2 style="color:white;text-align:center;">Prêt à dépenser </h2>
       </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)

    def get_pred_per_client():

        #filtering by client
        SK_ID_CURR = st.selectbox("Choisir un client", df['SK_ID_CURR'].unique())
        df_client = df[df['SK_ID_CURR'] == SK_ID_CURR]
        st.write(df_client)
        df_dico = dict(df_client.iloc[-1, :])
        st.success(f"L'identifient du client est: {SK_ID_CURR}")
        data = {
            'SK_ID_CURR': df_dico['SK_ID_CURR'],
            'YEARS_BIRTH': df_dico['YEARS_BIRTH'],
            'YEARS_EMPLOYED': df_dico['YEARS_EMPLOYED'],
            'AMT_ANNUITY': df_dico['AMT_ANNUITY'],
            'PAYMENT_RATE': df_dico['PAYMENT_RATE'],
            'INSTAL_DPD_MEAN': df_dico['INSTAL_DPD_MEAN'],
            'INSTAL_AMT_PAYMENT_SUM': df_dico['INSTAL_AMT_PAYMENT_SUM'],
            'PREV_CNT_PAYMENT_MEAN': df_dico['PREV_CNT_PAYMENT_MEAN'],
            'ACTIVE_DAYS_CREDIT_MAX': df_dico['ACTIVE_DAYS_CREDIT_MAX'],
            'APPROVED_CNT_PAYMENT_MEAN': df_dico['ACTIVE_DAYS_CREDIT_MAX'],
            'EXT_SOURCE_1': df_dico['EXT_SOURCE_1'],
            'EXT_SOURCE_2': df_dico['EXT_SOURCE_2'],
            'EXT_SOURCE_3': df_dico['EXT_SOURCE_3']
        }
        if st.button("Predict"):
            response = requests.post("https://vast-journey-10264.herokuapp.com/predict", json=data)
            prediction = response.text
            st.success(prediction)

        if st.button("Interpréter"):
            HtmlFile = open(r'lime_fig.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read()
            print(source_code)
            components.html(source_code, height=700)
    get_pred_per_client()

# Client page
if selected == "Prédiction des nouvaux clients":
    st.subheader("Renseignez les champs ci-dessous")
    def pred_new_client():
        SK_ID_CURR = st.number_input("Identifient du client", min_value=100001, value=100001, step=1)
        YEARS_BIRTH = st.number_input("L'âge du client",
                                      min_value=20, value=35, step=1)
        YEARS_EMPLOYED = st.number_input("Années d'expériences dans le poste actuel",
                                         min_value=0, value=15, step=1)
        AMT_ANNUITY = st.number_input("Le montant régulier du remboursement du prêt",
                                      min_value=1615, value=25000, step=100)
        PAYMENT_RATE = st.number_input("Le ratio du paiement du prêt",
                                       min_value=0.022, value=0.04, step=0.001)
        INSTAL_DPD_MEAN = st.number_input("La durée moyenne du retard du prêt",
                                          min_value=0, value=0, step=1)
        INSTAL_AMT_PAYMENT_SUM = st.number_input("La somme des paiements du remboursement des prêts",
                                                 min_value=0, value=0, step=1)
        PREV_CNT_PAYMENT_MEAN = st.number_input("Le paiement moyen du remboursement des prêts précents",
                                                min_value=0, value=300000, step=1000)
        ACTIVE_DAYS_CREDIT_MAX = st.number_input("Le nombre des jours maximums  ",
                                                 min_value=0, value=0, step=1)
        APPROVED_CNT_PAYMENT_MEAN = st.number_input("La durée moyenne du dernier prêt approuvé  ",
                                                    min_value=0, value=0, step=1)
        EXT_SOURCE_1 = st.slider("Le 1er score exterieur",
                                 min_value=0.014, max_value=0.9, value=0.5, step=0.001)
        EXT_SOURCE_2 = st.slider("Le 2nd score exterieur",
                                 min_value=0.014, max_value=0.9, value=0.5, step=0.001)
        EXT_SOURCE_3 = st.slider("Le 3e score exterieur",
                                 min_value=0.014, max_value=0.9, value=0.5, step=0.001)

        data = {
            'SK_ID_CURR': SK_ID_CURR,
            'YEARS_BIRTH': YEARS_BIRTH,
            'YEARS_EMPLOYED': YEARS_EMPLOYED,
            'AMT_ANNUITY': AMT_ANNUITY,
            'PAYMENT_RATE': PAYMENT_RATE,
            'INSTAL_DPD_MEAN': INSTAL_DPD_MEAN,
            'INSTAL_AMT_PAYMENT_SUM': INSTAL_AMT_PAYMENT_SUM,
            'PREV_CNT_PAYMENT_MEAN': PREV_CNT_PAYMENT_MEAN,
            'ACTIVE_DAYS_CREDIT_MAX': ACTIVE_DAYS_CREDIT_MAX,
            'APPROVED_CNT_PAYMENT_MEAN': APPROVED_CNT_PAYMENT_MEAN,
            'EXT_SOURCE_1': EXT_SOURCE_1,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'EXT_SOURCE_3': EXT_SOURCE_3
        }

        if st.button("Predict"):
            response = requests.post("https://vast-journey-10264.herokuapp.com/predict", json=data)
            prediction = response.text
            st.success(prediction)

    if __name__ == '__main__':
        # by default it will run at 8501 port
        pred_new_client()
# Comparison page
if selected == "Comparaison":
    st.title(f"Quelques comparaisons")

    """ # Regardons ensuite au travers d’une courbe la différence entre les valeurs prédites (bleu) et les valeurs attendues (rouge):
    y_train_predict = pd.DataFrame(model.predict(x_train))
    Y = y_train.copy()
    Y["Prediction"] = y_train_predict.to_numpy()
    Y.columns = ["Real", "Predict"]
    Y = Y.sort_index()
    Y["Id"] = Y.index
    Y["Delta"] = Y["Real"] - Y["Predict"]
    Y = Y.head(50)
    plt.rcParams["figure.figsize"] = (50, 10)
    Y["Real"].plot(color="#FF0000") # Red line
    Y["Predict"].plot(color="#0000FF") # Blue line
    """
