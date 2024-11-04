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
import requests
import feedparser
from sklearn.model_selection import train_test_split

#-----------------NASTAVENIA-----------------
st.set_page_config(layout="centered")
#------------------------------------------

st.title('Predikcia časových radov vybraných valutových kurzov')

def main():
    zobraz_spravy_v_sidebar()
    predikcia()

def stiahnut_data(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date, progress=False)
    # Flatten columns if necessary (handle multi-index case)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

moznost = st.selectbox('Zadajte menový tiker', ['EURUSD=X','EURCHF=X', 'EURAUD=X','EURNZD=X', 'EURCAD=X', 'EURSEK=X', 'EURNOK=X', 'EURCZK=X'])
moznost = moznost.upper()
dnes = datetime.date(2024, 10, 20)
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

# Calculating and plotting moving averages
# vypočet kĺzavého priemeru
st.header('Jednoduchý kĺzavý priemer za 50 dní')
datama50 = data.copy()
datama50['50ma'] = datama50['Close'].rolling(50).mean()
st.line_chart(datama50[['50ma', 'Close']])

st.header('Jednoduchý kĺzavý priemer za 200 dní')
datama200 = data.copy()
datama200['200ma'] = datama200['Close'].rolling(200).mean()
st.line_chart(datama200[['200ma', 'Close']])

# Merging 50ma and 200ma data for combined chart
spojene_data = pd.concat([datama200[['200ma', 'Close']], datama50[['50ma']]], axis=1)

st.header('Jednoduchý kĺzavý priemer za 50 dní a 200 dní')
st.line_chart(spojene_data)

def predikcia():
    model_options = {
        'Lineárna Regresia': LinearRegression(),
        'Regresor náhodného lesa': RandomForestRegressor(n_estimators=200),  # Zvýšený počet stromov
        'Regresor náhodného lesa': RandomForestRegressor(),
        'Regresor K najbližších susedov': KNeighborsRegressor()
    }
    
    model = st.selectbox('Vyberte model', list(model_options.keys()))
    pocet_dni = st.number_input('Koľko dní chcete predpovedať?', value=5)
    pocet_dni = int(pocet_dni)

    if st.button('Predikovať'):
        algoritmus = model_options.get(model)

        # Vytvorenie ďalších charakteristík (features)
        # Príprava dát pre "lagged" predikciu
        df = data[['Close']].copy()
        df['50ma'] = df['Close'].rolling(window=50).mean()
        df['200ma'] = df['Close'].rolling(window=200).mean()
        df['predikcia'] = df['Close'].shift(-pocet_dni)

        # Odstránenie riadkov s NaN hodnotami (vznikajúcimi z posunutia a kĺzavého priemeru)
        df.dropna(inplace=True)
        # Pridáme oneskorené hodnoty (lag features) ako prediktory
        for lag in range(1, pocet_dni + 1):
            df[f'lag_{lag}'] = df['Close'].shift(lag)

        # Vstupné a cieľové premenné
        x = df.drop(['predikcia'], axis=1).values
        y = df['predikcia'].values
        df.dropna(inplace=True)

        # Škálovanie dát
        x = scaler.fit_transform(x)
        x_predikcia = x[-pocet_dni:]
        x = x[:-pocet_dni]
        y = y[:-pocet_dni]
        # Vytvorenie vstupných a výstupných hodnôt
        x = df.drop(['Close'], axis=1).values
        y = df['Close'].values

        # Rozdelenie dát na trénovaciu a testovaciu množinu
        train_size = int(len(x) * 0.8)
        x_trenovanie, x_testovanie = x[:train_size], x[train_size:]
        y_trenovanie, y_testovanie = y[:train_size], y[train_size:]
        # Rozdelenie dát na 70 % trénovaciu a 30 % testovaciu množinu, bez miešania dát
        x_trenovanie, x_testovanie, y_trenovanie, y_testovanie = train_test_split(
            x, y, test_size=0.3, shuffle=False
        )

        # Trénovanie modelu
        algoritmus.fit(x_trenovanie, y_trenovanie)
        # Predikcia na testovacej množine
        predikcia = algoritmus.predict(x_testovanie)

        # Predikcia na základe počtu dní
        predikcia_forecast = algoritmus.predict(x_predikcia)
        # Výpočet metrík RMSE a MAE pre testovaciu množinu
        rmse = np.sqrt(mean_squared_error(y_testovanie, predikcia))
        mae = mean_absolute_error(y_testovanie, predikcia)
        st.text(f'RMSE: {rmse} \nMAE: {mae}')
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
        dnes = datetime.date.today()
        for i in predikcia_forecast:
            aktualny_datum = dnes + datetime.timedelta(days=den)
            st.text(f'Deň {den}: {i}')
            st.text(f'{aktualny_datum.strftime("%d. %B %Y")}: {i}')
            predikovane_data.append({'datum': aktualny_datum, 'predikcia': i})
            den += 1

        data_predicted = pd.DataFrame(predikovane_data)

        # Výpočet metrík pre hodnotenie modelu
        rmse = np.sqrt(np.mean((y_testovanie - predikcia) ** 2))
        mae = mean_absolute_error(y_testovanie, predikcia)
        st.text(f'RMSE: {rmse} \nMAE: {mae}')
        # Stiahnutie predikovaných dát ako CSV
        # Stiahnutie dát ako CSV
        csv = data_predicted.to_csv(index=False, sep=';', encoding='utf-8')
        st.download_button(
            label="Stiahnuť predikciu ako CSV",
            data=csv,
            file_name='predikcia.csv',
            file_name=f'predikcia_{moznost}.csv',
            mime='text/csv'
        )
        
def zobraz_spravy_v_sidebar():
    st.sidebar.header('Aktuálne Správy súvisiace s Menovým Trhom')
    st.sidebar.write('Načítavam aktuálne správy z RSS kanálov...')
    st.sidebar.header('Aktuálne Správy súvisiace s Menovým Trhom :dollar: ')
    st.sidebar.header('Aktuálne Správy súvisiace s Menovým Trhom :chart_with_upwards_trend:')
    st.sidebar.markdown('---')
    # Použitie RSS feedu pre načítanie finančných správ z Investing.com - Forex News sekcia
    feed_url = 'https://www.investing.com/rss/news_1.rss'  # RSS kanál zameraný na Forex News od Investing.com
    feed = feedparser.parse(feed_url)

    if len(feed.entries) > 0:
        for entry in feed.entries[:10]:  # Zobrazíme prvých 10 relevantných správ
        for entry in feed.entries[:15]: 
            st.sidebar.subheader(entry.title)
            if hasattr(entry, 'summary'):
                st.sidebar.write(entry.summary)
            st.sidebar.write(f"[\u010c\u00edta\u0165 viac]({entry.link})")
            st.sidebar.write(f"[Čítať viac]({entry.link})")
            st.sidebar.markdown('---')  # Pridanie oddeľovacej čiary medzi správami
    else:
        st.sidebar.write('Nenašli sa žiadne správy.')

if __name__ == '__main__':
    main()
