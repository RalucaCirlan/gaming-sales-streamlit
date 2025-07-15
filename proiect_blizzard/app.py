import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
import statsmodels.api as sm

# Configurare interfata
st.set_page_config(page_title="Analiza Blizzard", layout="wide")
st.title(" Analiza companiei Blizzard Entertainment")
st.markdown("Analizam activitatea si oportunitatile de extindere ale companiei Blizzard pe baza vanzarilor de jocuri.")

# 1. Incarcare fisier cu setul de date folosit
df = pd.read_csv("Games.csv")
df.columns = df.columns.str.replace('# ', '').str.strip()

# 2. Tratarea valorilor lipsa si extreme
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df['Sales'].fillna(df['Sales'].median(), inplace=True)
df['Sales'] = np.where(df['Sales'] > 45, 45, df['Sales'])

# 3. Codificare categorii (Genre)
le = LabelEncoder()
df['Genre_Code'] = le.fit_transform(df['Genre'].astype(str))


# 4. Extrage anul din data
def parse_year(release_str):
    try:
        return pd.to_datetime(release_str, format='%b-%y').year
    except:
        return np.nan

df['Year'] = df['Release'].apply(parse_year)
df['Year'].fillna(round(df['Year'].median()), inplace=True)


# 5. Scalare vanzari
scaler = MinMaxScaler()
df['Sales_Scaled'] = scaler.fit_transform(df[['Sales']])

# 6. Filtrare Blizzard
blizzard_df = df[df['Developer'] == "Blizzard Entertainment"]

# 7. Afisare date
st.subheader("Jocurile Blizzard din set")
st.dataframe(blizzard_df)

# 8. Grafice
st.subheader("Vanzari jocuri Blizzard")
fig1 = px.bar(blizzard_df, x='Name', y='Sales', color='Genre', title="Vanzari per joc")
st.plotly_chart(fig1)

# 9. Grupare/Aggregare
st.subheader("Vanzari medii pe gen (toti publisherii)")
sales_by_gen = df.groupby('Genre')['Sales'].mean().reset_index()
fig2 = px.pie(sales_by_gen, names='Genre', values='Sales', title="Distributia vanzarilor pe genuri")
st.plotly_chart(fig2)


# 10. Clusterizare doar pentru jocurile Blizzard
st.subheader("Clusterizare jocuri Blizzard (după vanzari + gen)")

cluster_data_blizzard = blizzard_df[['Sales_Scaled', 'Genre_Code']]
kmeans_blizzard = KMeans(n_clusters=3, random_state=42)
blizzard_df['Cluster'] = kmeans_blizzard.fit_predict(cluster_data_blizzard)

# Grafic scatter doar pentru Blizzard
fig3 = px.scatter(blizzard_df,
                  x='Sales_Scaled',
                  y='Genre_Code',
                  color=blizzard_df['Cluster'].astype(str),
                  hover_data=['Name', 'Genre'],
                  title="Clustere de jocuri Blizzard (după gen și vanzari)")
st.plotly_chart(fig3)

#  Clusterizare pt toate jocurile
st.subheader("Clusterizare jocuri (după vanzări + gen)")
cluster_data = df[['Sales_Scaled', 'Genre_Code']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(cluster_data)
fig3 = px.scatter(df, x='Sales_Scaled', y='Genre_Code', color=df['Cluster'].astype(str),
                  hover_data=['Name', 'Genre'], title="Clustere de jocuri")
st.plotly_chart(fig3)

# 11. Regresie
st.subheader("Regresie multipla: Vanzari - Gen + An")
X = df[['Genre_Code', 'Year']]
X = sm.add_constant(X)
y = df['Sales']
model = sm.OLS(y, X).fit()
st.text(model.summary())

# 12. Indicatori cheie
st.subheader(" Indicatori cheie")

# Total jocuri
total_games = len(df)
blizzard_games = len(blizzard_df)

# Medii
total_avg_sales = round(df['Sales'].mean(), 2)
blizzard_avg_sales = round(blizzard_df['Sales'].mean(), 2)

# Cel mai vândut joc Blizzard
top_blizzard_game = blizzard_df.loc[blizzard_df['Sales'].idxmax()]['Name']

# Afișare metrici
st.metric(label="Nr. total jocuri în set", value=total_games)
st.metric(label="Nr. jocuri Blizzard", value=blizzard_games)
st.metric(label="Vanzari medii (toți publisherii)", value=total_avg_sales)
st.metric(label="Vanzari medii Blizzard", value=blizzard_avg_sales)
st.metric(label="Cel mai vandut joc Blizzard", value=top_blizzard_game)


#13. Trend vanzari in timp
st.subheader("Evolutia vanzarilor Blizzard in timp")
trend = blizzard_df.groupby('Year')['Sales'].sum().reset_index()
fig_trend = px.line(trend, x='Year', y='Sales', markers=True, title="Trend vanzari Blizzard in timp")
st.plotly_chart(fig_trend)

# 14. Top Publisheri in funcyie de vanzarile medii
st.subheader("Comparatie Blizzard vs alti publisheri")

# Toti publisherii și mediile lor
all_publishers_avg = df.groupby('Publisher')['Sales'].mean().reset_index()

# Sortare pentru pozitionare completa (nu doar top 10)
all_publishers_sorted = all_publishers_avg.sort_values(by='Sales', ascending=False).reset_index(drop=True)

# Câți publisheri exista
total_publishers = len(all_publishers_sorted)

# Poziția reala a Blizzard
blizzard_position = all_publishers_sorted[all_publishers_sorted['Publisher'] == "Blizzard Entertainment"].index[0] + 1

# Calcul medie Blizzard
blizzard_avg = df[df['Publisher'] == "Blizzard Entertainment"]['Sales'].mean()

# Selectam doar top 10 pentru grafic
top_10_publishers = all_publishers_sorted.head(10)

# Grafic
fig_pub = px.bar(top_10_publishers, x='Publisher', y='Sales', title="Top 10 publisheri dupa vanzari medii")

# Linie orizontala pentru Blizzard
fig_pub.add_hline(y=blizzard_avg, line_dash="dash", line_color="red",
                  annotation_text="Media Blizzard", annotation_position="top right")

# Afisare
st.plotly_chart(fig_pub)

# Mesaj final informativ
st.info(f"Blizzard Entertainment se afla pe **locul {blizzard_position} din {total_publishers} publisheri** "
        f"in clasamentul vanzarilor medii per joc, cu o valoare de **{blizzard_avg:.2f} milioane**.")



st.markdown("---")
st.success("")

st.subheader(" Export set de date procesat")

csv = df.to_csv(index=False)
st.download_button(
    label="Descarca fișierul CSV",
    data=csv,
    file_name='games_processed.csv',
    mime='text/csv'
)
