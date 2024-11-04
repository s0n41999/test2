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

        # Príprava dát pre "lagged" predikciu - tu zmeníme počet lag features
        df = data[['Close']].copy()
        
        # Pridáme len zopár oneskorených hodnôt (napr. 1, 2, 3 dni)
        for lag in range(1, 4):
            df[f'lag_{lag}'] = df['Close'].shift(lag)

        df.dropna(inplace=True)

        # Vytvorenie vstupných a výstupných hodnôt
        x = df.drop(['Close'], axis=1).values
        y = df['Close'].values

        # Použitie train_test_split na rozdelenie dát (ako v prvom kóde)
        x_trenovanie, x_testovanie, y_trenovanie, y_testovanie = train_test_split(x, y, test_size=0.2, random_state=7)

        # Škálovanie vstupných dát
        x_trenovanie = scaler.fit_transform(x_trenovanie)
        x_testovanie = scaler.transform(x_testovanie)

        # Trénovanie modelu
        algoritmus.fit(x_trenovanie, y_trenovanie)

        # Predikcia na testovacej množine
        predikcia = algoritmus.predict(x_testovanie)

        # Predikcia budúcich hodnôt 
        posledne_data = x[-1].reshape(1, -1)
        predikcia_forecast = []
        
        for _ in range(pocet_dni):
            buduca_hodnota = algoritmus.predict(posledne_data)[0]
            predikcia_forecast.append(buduca_hodnota)
            
            # Aktualizujeme "lag" hodnoty na ďalšiu predikciu
            posledne_data = np.roll(posledne_data, -1)
            posledne_data[0, -1] = buduca_hodnota

        # Výpis predikcií
        den = 1
        predikovane_data = []
        for i in predikcia_forecast:
            aktualny_datum = dnes + datetime.timedelta(days=den)
            st.text(f'{aktualny_datum.strftime("%d. %B %Y")}: {i}')
            predikovane_data.append({'datum': aktualny_datum, 'predikcia': i})
            den += 1

        data_predicted = pd.DataFrame(predikovane_data)

        # Hodnotenie modelu
        rmse = np.sqrt(np.mean((y_testovanie - predikcia) ** 2))
        mae = mean_absolute_error(y_testovanie, predikcia)
        st.text(f'RMSE: {rmse} \nMAE: {mae}')

if __name__ == '__main__':
    main()
