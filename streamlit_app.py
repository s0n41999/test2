import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error  

#-----------------NASTAVENIA-----------------

st.set_page_config(layout="centered")

st.title('Predikcia časových radov vybraných valutových kurzov')

def main():
    predikcia()

def stiahnut_data(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date, progress=False)
    # Flatten columns if necessary (handle multi-index case)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

moznost = st.selectbox('Zadajte menový tiker', [
    'EURUSD=X', 'EURCHF=X', 'EURAUD=X', 'EURNZD=X', 'EURCAD=X', 
    'EURSEK=X', 'EURNOK=X', 'EURCZK=X', 'USDTRY=X', 'GBPJPY=X', 'USDZAR=X', 
    'USDMXN=X', 'USDBRL=X', 'USDRUB=X', 'AUDNZD=X', 'USDINR=X'
])
moznost = moznost.upper()
dnes = datetime.date.today()
start = dnes - datetime.timedelta(days=3650)
start_date = start
end_date = dnes

data = stiahnut_data(moznost, start_date, end_date)
scaler = StandardScaler()

# Ensure the 'Close' column is accessible even if multi-indexed
close_column = [col for col in data.columns if 'Close' in col]
if close_column:
    data['Close'] = data[close_column[0]]

# Plotting the data
st.write('Záverečný kurz')
st.line_chart(data['Close'])
st.header('Nedávne Dáta')
st.dataframe(data.tail(20))

def predikcia():
    model_options = {
        'Lineárna Regresia': LinearRegression(),
        'Regresor náhodného lesa': RandomForestRegressor(),
        'Regresor K najbližších susedov': KNeighborsRegressor()
    }
    
    model = st.selectbox('Vyberte model', list(model_options.keys()))
    pocet_dni = st.number_input('Koľko dní chcete predpovedať?', value=5)
    pocet_dni = int(pocet_dni)
    
    if st.button('Predikovať'):
        algoritmus = model_options.get(model)

        # Príprava dát
        df = data[['Close']].copy()

        # Presunúť 'Close' do y a vytvoriť oneskorené vstupy pre x
        df['predikcia'] = df['Close'].shift(-pocet_dni)
        df.dropna(inplace=True)

        x = df.drop(['predikcia'], axis=1).values
        y = df['predikcia'].values

        # Škálovanie vstupných dát
        x = scaler.fit_transform(x)

        # Rozdelenie dát na tréning a testovaciu množinu pomocou train_test_split
        x_trenovanie, x_testovanie, y_trenovanie, y_testovanie = train_test_split(x, y, test_size=0.2, random_state=7)

        # Trénovanie modelu
        algoritmus.fit(x_trenovanie, y_trenovanie)

        # Predikcia na testovacej množine
        predikcia = algoritmus.predict(x_testovanie)

        # Predikcia na základe budúcich dní
        x_predikcia = x[-pocet_dni:]
        predikcia_forecast = algoritmus.predict(x_predikcia)
        
        # Výpis predikcií
        den = 1
        predikovane_data = []
        col1, col2 = st.columns(2)

        with col1:
            for i in predikcia_forecast:
                aktualny_datum = dnes + datetime.timedelta(days=den)
                st.text(f'Deň {den}: {i}')
                predikovane_data.append({'Deň': aktualny_datum, 'Predikcia': i})
                den += 1

        with col2:
            data_predicted = pd.DataFrame(predikovane_data)
            st.dataframe(data_predicted)

        # Hodnotenie modelu
        rmse = np.sqrt(np.mean((y_testovanie - predikcia) ** 2))
        mae = mean_absolute_error(y_testovanie, predikcia)
        st.text(f'RMSE: {rmse} \nMAE: {mae}')

if __name__ == '__main__':
    main()
