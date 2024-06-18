import ast

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import cv2


# df_all = pd.read_csv(r'D:\Project-mpg microgesture\human_micro_gesture_classifier\experiments\resampled_64_multi_update_freq_8\embedded_data_clipped.csv')
df_all = pd.read_csv(r'D:\Project-mpg microgesture\human_micro_gesture_classifier\experiments\resampled_64_multi_update_freq_8\embedded_data_expanded.csv')
multi_labels = True


if multi_labels:
    df_all_sub = df_all.loc[df_all['label_multi'].apply(lambda x: sum(ast.literal_eval(x.replace(' ',','))) == 1 or np.any(np.array(ast.literal_eval(x.replace(' ',',')))[[0,3]]==1))]
else:
    df_all_sub = df_all


# Create the Dash app with Flask server
app = dash.Dash(__name__)
server = app.server


def display_video(file_path, scale_factor):
    # print(f'{scale_factor=}')
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return html.Div('Error: Cannot open video file', style={'color': 'red'})

    ret, frame = cap.read()
    height, width, _ = frame.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', new_width, new_height)

    while ret:
        resized_frame = cv2.resize(frame, (new_width, new_height))
        cv2.imshow('Video', resized_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

app.layout = html.Div([
    dcc.Graph(
        id='scatter-plot',
        figure=px.scatter(
            df_all_sub,
            x='emb1',
            y='emb2',
            facet_col='sample',
            color='label',
            title='Scatter plot of emb1 vs emb2 faceted by sample and colored by label',
            hover_data=['file_names', 'file_paths']
        ).update_layout(
            margin=dict(l=20, r=20, t=40, b=20), height=800, width=1600
        )
    ),
    html.Div(id='click-output')
])

@app.callback(
    Output('click-output', 'children'),
    Input('scatter-plot', 'clickData')
)
def display_click_data(clickData):
    if clickData:
        point_info = clickData['points'][0]
        file_path = point_info['customdata'][1]  # Assuming 'file_paths' is the second in customdata
        display_video(file_path, scale_factor=0.5)
        return html.Pre(str(point_info))
    return "Click on a point to see its information here."

if __name__ == '__main__':
    app.run_server(debug=True)
