# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:01:18 2025

@author: fabia
"""

import requests

def upload_training_data(
    url,
    model_file_path,
    graph_image_paths,
    episode,
    reward,
    loss,
    profit_usd,
    epsilon,
    drawdown,
    hit_frequency
):
    data = {
        'episode': (None, str(episode)),
        'reward': (None, str(reward)),
        'loss': (None, str(loss)),
        'profitUSD': (None, str(profit_usd)),
        'epsilon': (None, str(epsilon)),
        'drawdown': (None, str(drawdown)),
        'hitFrequency': (None, str(hit_frequency)),
    }

    files = {}

    open_files = []  # Para cerrar todo al final

    try:
        if model_file_path and model_file_path != "no":
            f = open(model_file_path, 'rb')
            files['modelFile'] = (model_file_path.split('/')[-1], f)
            open_files.append(f)

        if graph_image_paths:
            for i, img_path in enumerate(graph_image_paths):
                f = open(img_path, 'rb')
                files['graphImages'] = files.get('graphImages', [])
                files['graphImages'].append((img_path.split('/')[-1], f))
                open_files.append(f)

        # requests requiere una estructura especial para múltiples archivos con el mismo nombre
        # Si hay múltiples graphImages, debemos transformar 'graphImages' en lista de tuplas
        if 'graphImages' in files:
            graph_files = files.pop('graphImages')
            for idx, (filename, fileobj) in enumerate(graph_files):
                files['graphImages'] = graph_files if len(graph_files) == 1 else [
                    ('graphImages', (filename, fileobj)) for (filename, fileobj) in graph_files
                ]
                break  # ya transformamos todo

        response = requests.post(url, data=data, files=files)
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {'raw_response': response.text}

    except Exception as e:
        return 500, {'error': str(e)}

    finally:
        for f in open_files:
            try:
                f.close()
            except:
                pass
