import pandas as pd 
import plotly.express as px 
from dash import Dash, dcc, html, Input, Output
 
df = pd.read_excel(r'C:\Users\Test\Desktop\DEPI-ONL4_AIS2_S2\dataAnalysis\session3\Dash.xlsx')
app = Dash()
app.title = " Interactive Dashboard"
num_cols = df.select_dtypes(include='number').columns
 
app.layout = html.Div([html.H1("interactive dashboard with  pie  chart"),
                      html.Label("select a value to show in the pie chart"),
                      dcc.Dropdown(id = 'column-dropdown', options=[{'label':col,'value':col}for col in num_cols],
                                   value=num_cols[0]),
                      dcc.Graph(id = 'pie-chart')
                      ])
@app.callback(Output('pie-chart', 'figure'),
              Input('column-dropdown','value'))
def update_pie(selected_col):
    grouped = df.groupby('Area')[selected_col].sum().reset_index()
    fig = px.pie(grouped,names='Area',values=selected_col, title=f"Distribution of {selected_col} by Area",hole=0.4,
                 color_discrete_sequence= px.colors.qualitative.Set2)

    return fig
 
if __name__ == "__main__":
    app.run(debug = True)