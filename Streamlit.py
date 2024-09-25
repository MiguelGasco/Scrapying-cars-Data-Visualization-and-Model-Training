import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


def main():
    # Configuración de la página
    df = pd.read_csv('cars_Cleaned.csv')
    st.set_page_config(page_title="Mi Aplicación", layout="wide")

    # Crear el sidebar
    with st.sidebar:
        # Crear las opciones del sidebar
        opcion = st.radio(
            "",
            ("Data Visualization", "Model")
        )

    # Contenido principal basado en la selección del sidebar
    if opcion == "Data Visualization":
        st.header("Data Visualization")
        #############
        #Correlation#
        #############
        st.header('Correlation heatmap')

        corr_matrix = df.iloc[:, :10].corr()

        fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdBu',
                        zmin=-1, zmax=1,
                        text=corr_matrix.values.round(2),
                        texttemplate='%{text}',
                        textfont={"size":10},
                        hoverongaps = False))

        fig.update_layout(
            width=800,
            height=700
        )

        st.plotly_chart(fig, use_container_width=True)

        ###########
        #Histogram#
        ###########
        st.header('Histogram first 10 columns')

        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns[:10]

        fig = make_subplots(rows=4, cols=3, subplot_titles=numeric_columns)

        for i, col in enumerate(numeric_columns, 1):
            row = (i - 1) // 3 + 1
            col_num = (i - 1) % 3 + 1
            
            fig.add_trace(
                go.Histogram(x=df[col], name=col),
                row=row, col=col_num
            )

        fig.update_layout(
            height=800,
            width=1000,
            showlegend=False,
        )

        fig.update_layout(margin=dict(l=50, r=50, t=50, b=50), bargap=0.1)

        st.plotly_chart(fig, use_container_width=True)

        #########
        #Scatter#
        #########
        st.header('Price vs. Mileage by Fuel Type')
        df_copy = df.copy()

        fuel_map = {1: 'Gasolina', 0: 'Diesel'}
        df_copy['Fuel'] = df_copy['Fuel'].map(fuel_map)

        fig = px.scatter(df_copy, x='Mileage', y='price', color='Fuel', hover_data='power')

        fuel_types = df_copy['Fuel'].unique()
        buttons = []

        df_copy['Fuel'] = df_copy['Fuel'].astype(str)
        buttons.append(dict(
            args=[{'visible': [True] * len(fig.data)}],
            label='Todos',
            method='update'
        ))

        for fuel in fuel_types:
            buttons.append(dict(
                args=[{'visible': [fuel == trace.name for trace in fig.data]}],
                label=fuel,
                method='update'
            ))

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="right",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1.02,
                xanchor="left",
                y=0.8,
                yanchor="top"
            )]
        )

        fig.update_layout(
            xaxis_title='Kilometraje',
            yaxis_title='Precio',
            legend_title='Tipo de Combustible',
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)

        #################
        #Bar Confort#####
        #################
        st.header('Top 10 Comfort Features That Most Influence Price')
        comfort_columns = [col for col in df_copy.columns if col.startswith('Comfort_')]

        average_price = df_copy['price'].mean()

        comfort_prices = []
        for col in comfort_columns:
            avg_price_with_feature = df_copy[df_copy[col] == 1]['price'].mean()
            price_difference = avg_price_with_feature - average_price
            comfort_prices.append((col.replace('Comfort_', ''), price_difference))

        top_10_comfort = sorted(comfort_prices, key=lambda x: x[1], reverse=True)[:10]

        chart_data = pd.DataFrame(top_10_comfort, columns=['Característica', 'Diferencia de Precio'])

        fig = px.bar(chart_data, x='Característica', y='Diferencia de Precio',
                    labels={'Característica': 'Confort', 'Diferencia de Precio': 'Diferencia de Precio (€)'},
                    color='Diferencia de Precio')

        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            yaxis_title='Diferencia de Precio (€)'
        )

        st.plotly_chart(fig, use_container_width=True)

        #######################
        #Bar Entertainment#####
        #######################
        st.header('Top 10 Entertainment Features That Most Influence Price')
        entertainment_columns = [col for col in df.columns if col.startswith('Entertainment_')]

        average_price = df['price'].mean()

        entertainment_prices = []
        for col in entertainment_columns:
            avg_price_with_feature = df[df[col] == 1]['price'].mean()
            price_difference = avg_price_with_feature - average_price
            entertainment_prices.append((col.replace('Entertainment_', ''), price_difference))

        top_10_entertainment = sorted(entertainment_prices, key=lambda x: x[1], reverse=True)[:10]

        chart_data = pd.DataFrame(top_10_entertainment, columns=['Característica', 'Diferencia de Precio'])

        fig = px.bar(chart_data, x='Característica', y='Diferencia de Precio',
                    labels={'Característica': 'Entertainment', 'Diferencia de Precio': 'Diferencia de Precio (€)'},
                    color='Diferencia de Precio')


        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            yaxis_title='Diferencia de Precio (€)'
        )

        st.plotly_chart(fig, use_container_width=True)

        #######################
        #Bar Extras############
        #######################
        st.header('Top 10 Extras Features That Most Influence Price')
        extras_columns = [col for col in df.columns if col.startswith('Extras_')]

        average_price = df['price'].mean()

        extras_prices = []
        for col in extras_columns:
            avg_price_with_feature = df[df[col] == 1]['price'].mean()
            price_difference = avg_price_with_feature - average_price
            extras_prices.append((col.replace('Extras_', ''), price_difference))

        top_10_extras = sorted(extras_prices, key=lambda x: x[1], reverse=True)[:10]

        chart_data = pd.DataFrame(top_10_extras, columns=['Característica', 'Diferencia de Precio'])

        fig = px.bar(chart_data, x='Característica', y='Diferencia de Precio',
                    labels={'Característica': 'Extras', 'Diferencia de Precio': 'Diferencia de Precio (€)'},
                    color='Diferencia de Precio')

        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            yaxis_title='Diferencia de Precio (€)'
        )

        st.plotly_chart(fig, use_container_width=True)

        #######################
        #Bar Security##########
        #######################
        st.header('Top 10 Security Features That Most Influence Price')
        security_columns = [col for col in df.columns if col.startswith('Security_')]

        average_price = df['price'].mean()

        security_prices = []
        for col in security_columns:
            avg_price_with_feature = df[df[col] == 1]['price'].mean()
            price_difference = avg_price_with_feature - average_price
            security_prices.append((col.replace('Security_', ''), price_difference))


        top_10_comfort = sorted(security_prices, key=lambda x: x[1], reverse=True)[:10]


        chart_data = pd.DataFrame(top_10_extras, columns=['Característica', 'Diferencia de Precio'])

        fig = px.bar(chart_data, x='Característica', y='Diferencia de Precio',
                    labels={'Característica': 'Security', 'Diferencia de Precio': 'Diferencia de Precio (€)'},
                    color='Diferencia de Precio')

        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            yaxis_title='Diferencia de Precio (€)'
        )

        # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)


        #######################
        #Pie Transmission#####
        #######################
        st.header('Number of Cars by Transmission')
        df_copy = df.copy()

        gearbox_map = {0: 'Automático', 1: 'Manual'}
        df_copy['Gearbox'] = df_copy['Gearbox'].map(gearbox_map)
        gearbox_counts = df_copy['Gearbox'].value_counts()
        
        fig = px.pie(
            values=gearbox_counts.values,
            names=gearbox_counts.index,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

        ###############
        #Bar Power#####
        ###############
        st.header('Vehicle Power Distribution')
        fig = px.histogram(df, x='power', 
                        nbins=5,  # Ajusta el número de bins según sea necesario
                        labels={'power': 'Potencia (HP)', 'count': 'Número de Vehículos'}
                        )

        fig.update_layout(
            xaxis_title='Potencia (HP)',
            yaxis_title='Número de Vehículos',
            bargap=0.1 
        )
        bins = st.slider('Número de bins', min_value=20, max_value=30, value=20, step=1)

        # Actualizar el gráfico con el nuevo número de bins
        fig.update_traces(nbinsx=bins)

        # Volver a mostrar el gráfico actualizado
        st.plotly_chart(fig, use_container_width=True)


        #########
        #Lip#####
        #########
        st.header('Price Distribution by Fuel Type')
        df_copy = df.copy()
        fuel_map = {0: 'Gasolina', 1: 'Diesel'}
        df_copy['Fuel'] = df_copy['Fuel'].map(fuel_map)

        # Crear el gráfico de caja y bigotes
        fig = px.box(df_copy, x='Fuel', y='price', 
                    labels={'Fuel': 'Tipo de Combustible', 'price': 'Precio (€)'},
                    color='Fuel',
                    points="all",
                    category_orders={'Fuel': ['Gasolina', 'Diesel']})

        # Personalizar el diseño
        fig.update_layout(
            xaxis_title='Tipo de Combustible',
            yaxis_title='Precio (€)',
            showlegend=False,
            height = 800
        )

        st.plotly_chart(fig, use_container_width=True)

        ################
        #Scatter 3D#####
        ################
        st.header('Price vs. Mileage vs. Power')
        df_copy = df.copy()
        fuel_map = {0: 'Gasolina', 1: 'Diesel'}
        df_copy['Fuel'] = df_copy['Fuel'].map(fuel_map)

        fig = px.scatter_3d(df_copy, x='Mileage', y='power', z='price',
                            color='Fuel',
                            labels={'Mileage': 'Kilometraje', 
                                    'power': 'Potencia (HP)', 
                                    'price': 'Precio (€)',
                                    'Fuel': 'Combustible'},)

        fig.update_layout(scene = dict(
                            xaxis_title='Kilometraje',
                            yaxis_title='Potencia (HP)',
                            zaxis_title='Precio (€)'),
                        width=900, height=700,
                        margin=dict(r=20, b=10, l=10, t=40))

        st.plotly_chart(fig, use_container_width=True)

    elif opcion == "Model":
        st.header("Model")
        st.header("Linear Regression Model Predictions")
        selected_features_rfe = pd.Index(['Gearbox', 'Comfort_Electrical window lifter',
       'Comfort_Electric tailgate', 'Comfort_Electric side mirrors',
       'Comfort_Start/Stop automatic', 'Comfort_Foldable passenger seat',
       'Comfort_Wind Deflector(for convertible)', 'Comfort_partial rear seat',
       'Entertainment_Bluetooth', 'Entertainment_CD',
       'Entertainment_DAB radio', 'Entertainment_TV',
       'Extras_Alloy rims (18")', 'Extras_Electronic parking brake',
       'Extras_Awning', 'Extras_Paddle shifters', 'Extras_Sports suspension',
       'Security_Passenger airbag', 'Security_Isofix',
       'Security_LED headlights', 'Security_LED daytime running lights',
       'Security_Side_Airbag', 'Security_Power Steering',
       'Security_Traction Control', 'Security_Xenon headlights', 'Mileage',
       'Owners', 'power'])

        X = df[selected_features_rfe]
        y = df['price']
        print(X.shape)
        print(y.shape)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        sc = StandardScaler()

        X_train_std = sc.fit_transform(X_train)
        X_test_std = sc.transform(X_test)

        linear_model = LinearRegression()
        linear_model.fit(X_train_std,y_train)
        y_pred = linear_model.predict(X_test_std)


        plot_data = pd.DataFrame({'Precio real': y_test, 'Precio predicho': y_pred})


        fig = px.scatter(plot_data, x='Precio real', y='Precio predicho', 
                        labels={'Precio real': 'Precio real (€)', 
                                'Precio predicho': 'Precio predicho (€)'}
                        )


        fig.add_trace(px.line(x=[plot_data['Precio real'].min(), plot_data['Precio real'].max()], 
                            y=[plot_data['Precio real'].min(), plot_data['Precio real'].max()]).data[0])

    
        fig.update_layout(
            width=800, 
            height=600,
            showlegend=False
        )

        
        r2 = linear_model.score(X_test_std, y_test)
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.05, y=0.95,
            text=f'R² = {r2:.2f}',
            showarrow=False,
            font=dict(size=14)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.header("Error Summary")
        residuos = y_test - y_pred
        df_residuos = pd.DataFrame({
            'Precio real': y_test,
            'Residuos': residuos
        })

        fig = px.scatter(df_residuos, x='Precio real', y='Residuos',
                 labels={'Precio real': 'Precio real (€)', 
                         'Residuos': 'Error (precio real - precio predicho) (€)'}
                 )
        
        fig.add_hline(y=0, line_dash="dash", line_color="#83c9ff")

        fig.update_layout(
            width=800, 
            height=600,
            xaxis_title='Precio real (€)',
            yaxis_title='Error (precio real - precio predicho) (€)'
        )

        st.plotly_chart(fig, use_container_width=True)
if __name__ == "__main__":
    main()