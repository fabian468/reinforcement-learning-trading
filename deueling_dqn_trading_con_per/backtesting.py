
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from indicadores import  rsi , macd , add_ema200_distance

from model_pytorch import DuelingDQN

class Backtesting_model:
    def __init__(
        self,
        model_name="AITrader",
        random_market_event_probability=0.01,
        spread=0.20,
        commission_per_trade=0.07
    ):
        self.inventory = []
        self.inventory_sell = []
        self.model_name = model_name
        self.random_market_event_probability = random_market_event_probability
        self.spread = spread
        self.commission_per_trade = commission_per_trade
        

        self.profit_history = []
        self.rewards_history = []
        self.epsilon_history = []
        self.trades_history = []
        self.drawdown_history = []
        self.sharpe_ratios = []
        self.accuracy_history = []
        self.avg_win_history = []
        self.avg_loss_history = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Asegúrate de definir el modelo antes
        self.model =  DuelingDQN(60, 5).to(self.device)

    def trade(self, state):
        with torch.no_grad():
           state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
           q_values = self.model(state_tensor)
           return torch.argmax(q_values, dim=1).item()
       
    def load_model(self, name):
        try:
            # Cargar checkpoint
            checkpoint = torch.load(f"{name}.pth", map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
    
            print(f"Modelo cargado desde {name}.pth")
    
        except Exception as e:
            print(f"Error al cargar modelo: {e}")
            print("Manteniendo valores por defecto.")

    def plot_training_metrics(self, save_path='resultados_cv'):
        min_length = min(len(self.profit_history), len(self.rewards_history), 
                        len(self.epsilon_history), len(self.trades_history), 
                        len(self.loss_history), len(self.drawdown_history), 
                        len(self.sharpe_ratios), len(self.accuracy_history), 
                        len(self.avg_win_history), len(self.avg_loss_history),
                        len(self.lr_history))

        episodes = range(1, min_length + 1)

        fig, axs = plt.subplots(5, 2, figsize=(15, 20))  # 5 filas, 2 columnas
        fig.suptitle('Métricas de Entrenamiento', fontsize=16)

        axs[0, 0].plot(episodes, self.profit_history[:min_length], label='Beneficio Total')
        axs[0, 0].set_ylabel('Beneficio')
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[0, 1].plot(episodes, self.epsilon_history[:min_length], label='Epsilon', color='red')
        axs[0, 1].set_ylabel('Epsilon')
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        axs[1, 0].plot(episodes, self.loss_history[:min_length], label='Loss', color='purple')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        axs[1, 1].plot(episodes, self.drawdown_history[:min_length], label='Drawdown Máximo', color='orange')
        axs[1, 1].set_ylabel('Drawdown')
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        axs[2, 0].plot(episodes, self.sharpe_ratios[:min_length], label='Ratio de Sharpe', color='green')
        axs[2, 0].set_ylabel('Ratio de Sharpe')
        axs[2, 0].grid(True)
        axs[2, 0].legend()

        axs[2, 1].plot(episodes, self.accuracy_history[:min_length], label='Frecuencia de Aciertos', color='brown')
        axs[2, 1].set_ylabel('Frecuencia de Aciertos')
        axs[2, 1].grid(True)
        axs[2, 1].legend()

        axs[3, 0].plot(episodes, self.rewards_history[:min_length], label='Recompensa por Episodio', color='blue')
        axs[3, 0].set_ylabel('Recompensa')
        axs[3, 0].grid(True)
        axs[3, 0].legend()

        # Nueva gráfica: Learning Rate
        axs[3, 1].plot(episodes, self.lr_history[:min_length], label='Learning Rate', color='magenta')
        axs[3, 1].set_ylabel('Learning Rate')
        axs[3, 1].set_xlabel('Episodio')
        axs[3, 1].grid(True)
        axs[3, 1].legend()
        axs[3, 1].set_yscale('log')  # Escala logarítmica para mejor visualización

        # Gráfica combinada: Loss vs Learning Rate
        axs[4, 0].plot(episodes, self.loss_history[:min_length], label='Loss', color='purple', alpha=0.7)
        axs[4, 0].set_ylabel('Loss', color='purple')
        axs[4, 0].tick_params(axis='y', labelcolor='purple')
        
        ax2 = axs[4, 0].twinx()
        ax2.plot(episodes, self.lr_history[:min_length], label='Learning Rate', color='magenta', alpha=0.7)
        ax2.set_ylabel('Learning Rate', color='magenta')
        ax2.tick_params(axis='y', labelcolor='magenta')
        ax2.set_yscale('log')
        
        axs[4, 0].set_xlabel('Episodio')
        axs[4, 0].grid(True)
        axs[4, 0].set_title('Loss vs Learning Rate')

        axs[4, 1].axis('off') # Para la celda vacía en la última fila

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_path, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()



def state_creator_vectorized(data, timestep, window_size):
    tiempo_rsi = 12
    tiempo_macd = 12
    tiempo_ema = 200
    
    # Para el state, usamos solo la ventana especificada
    starting_id = timestep - window_size + 1
    if starting_id < 0:
        padding = pd.DataFrame([data.iloc[0]] * -starting_id, columns=data.columns, index=data.index[: -starting_id])
        windowed_data = pd.concat([padding, data.iloc[0:timestep+1]])
    else:
        windowed_data = data.iloc[starting_id:timestep+1]

    # Escalado de precios y volumen para la ventana
    close_scaled = StandardScaler().fit_transform(windowed_data['close'].values.reshape(-1, 1)).flatten()
    volume_scaled = StandardScaler().fit_transform(windowed_data['tick_volume'].values.reshape(-1, 1)).flatten()
    
    state = []
    state.extend(close_scaled[1:] - close_scaled[:-1])
    state.extend(volume_scaled[1:] - volume_scaled[:-1])


    # RSI: usar datos desde (timestep - tiempo_rsi) hasta timestep
    if timestep >= tiempo_rsi - 1:  # Solo calcular si tenemos suficientes datos
        rsi_start = max(0, timestep - tiempo_rsi)
        rsi_data = data.iloc[rsi_start:timestep+1]
        
        try:
            rsi_values = rsi(rsi_data, period=tiempo_rsi).dropna().values
            if len(rsi_values) >= window_size:
                rsi_to_add = rsi_values[-window_size:] / 100.0
            elif len(rsi_values) > 0:
                padding_needed = window_size - len(rsi_values)
                rsi_to_add = np.concatenate([[0.5] * padding_needed, rsi_values / 100.0])
            else:
                rsi_to_add = [0.5] * window_size
                
        except Exception as e:
            if timestep >= 194:
                print(f"ERROR in RSI calculation: {e}")
            rsi_to_add = [0.5] * window_size
    else:
        rsi_to_add = [0.5] * window_size
    
    state.extend(rsi_to_add)
 

    # MACD: usar datos desde (timestep - tiempo_macd) hasta timestep
    if timestep >= tiempo_macd - 1:  # Solo calcular si tenemos suficientes datos
        macd_start = max(0, timestep - tiempo_macd + 1)
        macd_data = data.iloc[macd_start:timestep+1]
        
        try:
            macd_line, signal_line = macd(macd_data)
            macd_diff = (macd_line - signal_line).dropna().values
            
            if len(macd_diff) >= window_size:
                macd_to_add = macd_diff[-window_size:] / 10.0
            elif len(macd_diff) > 0:
                padding_needed = window_size - len(macd_diff)
                macd_to_add = np.concatenate([[0.0] * padding_needed, macd_diff / 10.0])
            else:
                macd_to_add = [0.0] * window_size
                
        except Exception as e:
            if timestep >= 194:
                print(f"ERROR in MACD calculation: {e}")
            macd_to_add = [0.0] * window_size
    else:
        macd_to_add = [0.0] * window_size

    state.extend(macd_to_add)
    
    # Información temporal
    time_str = data.iloc[timestep]['time']
    hour = int(time_str.split(':')[0])
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    state.extend([hour_sin, hour_cos])
    

    # EMA200: usar datos desde (timestep - tiempo_ema) hasta timestep
    if timestep >= tiempo_ema - 1:
        ema_start = max(0, timestep - tiempo_ema + 1)
        ema_data = data.iloc[ema_start:timestep+1]
    else:
        ema_data = data.iloc[0:timestep+1]  # Usar todos los datos disponibles
    
    try:
        dist_ema200, ema_200 = add_ema200_distance(ema_data)
        
        # Tomar los últimos valores correspondientes al window_size
        if len(dist_ema200) >= window_size:
            dist_ema_window = dist_ema200[-window_size:].fillna(0).values
            ema_200_window = ema_200[-window_size:].bfill().fillna(0).values
        else:
            available_dist = dist_ema200.fillna(0).values
            available_ema = ema_200.bfill().fillna(0).values
            
            padding_needed = window_size - len(available_dist)
            dist_ema_window = np.concatenate([np.zeros(padding_needed), available_dist])
            ema_200_window = np.concatenate([np.zeros(padding_needed), available_ema])
        
        # Escalar los valores de EMA
        dist_scaled = StandardScaler().fit_transform(dist_ema_window.reshape(-1, 1)).flatten()
        ema_scaled = StandardScaler().fit_transform(ema_200_window.reshape(-1, 1)).flatten()
        
    except Exception as e:
        if timestep >= 194:
            print(f"ERROR in EMA calculation: {e}")
        dist_scaled = np.zeros(window_size)
        ema_scaled = np.zeros(window_size)
    
    state.extend(dist_scaled)
    state.extend(ema_scaled)

    return np.array(state).reshape(1, -1)



def dataset_loader_csv(csv_path):
    try:
        df = pd.read_csv(f"data/{csv_path}", sep='\t')  # Usa tabulador como separador
        df.columns = [col.strip('<>').lower() for col in df.columns]  # Limpia < > y normaliza nombres
        ema_200_diferencia, _ = add_ema200_distance(df)
        # Verifica si existen las columnas necesarias
        
        if 'date' not in df.columns:
            print("Error: La columna 'date' es obligatoria.")
            return None
        if len(ema_200_diferencia) > 0:
            df['ema_diference_close'] = ema_200_diferencia
        # Combina fecha y hora si existe la columna 'time'
        if 'time' in df.columns and df['time'].notnull().all():
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'].astype(str))
        else:
            print("Advertencia: No se encontró columna 'time' o tiene valores nulos. Se usará solo 'date'.")
            df['datetime'] = pd.to_datetime(df['date'])

        df.set_index('datetime', inplace=True)

        # Renombra columna de volumen si existe
        if 'tickvol' in df.columns:
            df.rename(columns={'tickvol': 'tick_volume'}, inplace=True)

        print(f"Datos cargados desde: {csv_path}")

        # Selecciona columnas disponibles
        if 'time' in df.columns and 'close' in df.columns and 'spread' in df.columns and 'tick_volume' in df.columns:
            return df[['time', 'close', 'tick_volume' , 'spread' , 'low' , 'ema_diference_close']]
        elif 'close' in df.columns and 'tick_volume' in df.columns:
            return df[['close', 'tick_volume']]
        elif 'close' in df.columns:
            print("Advertencia: La columna 'tick_volume' no se encontró. Se usará tickvol=0.")
            df['tickvol'] = 0
            return df[['close', 'tickvol']]
        else:
            print("Error: El archivo CSV debe contener al menos la columna 'close'.")
            return None

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: {csv_path}")
        return None
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return None

def price_format(n):
    n = float(n)
    return "- {0:.3f}".format(abs(n)) if n < 0 else "{0:.3f}".format(abs(n))


def main():
    
    nombre_csv = "XAUUSD_H1_2015_01_01_2024_05_31.csv"
       
    
    cargar_modelo = False
    modelo_existente = "resultados_cv/model_XAUUSD_H1_2015_01_01_2024_05_31.csv"
    
    cargar_memoria_buffer = True

    es_indice = False
    es_forex = False
    es_metal = True
    tick_value = 5  
    pip_multiplier = 10000  
    

    balance_first = 100 # dinero inicial
    lot_size = 0.01   
    commission_per_trade = 4.5
    
    pip_value_eur_usd = 10 * lot_size
    
    window_size = 10
    
    # Creación de carpeta para guardar los resultados
    resultados_dir = 'resultados_cv'
    os.makedirs(resultados_dir, exist_ok=True)
    

    data = dataset_loader_csv(nombre_csv)
            
    if 'time' in data.columns and data['time'].notnull().all():
        hora = data['time']

    
    # Carga el modelo y ve si cargar uno o crear uno nuevo
    trader = Backtesting_model(
                       commission_per_trade= commission_per_trade,
                       )

    if cargar_modelo:
        try:
            trader.load_model(modelo_existente , cargar_memoria_buffer)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar el modelo {modelo_existente}: {str(e)}")
    
    data_samples = len(data) - 1 


    states =[state_creator_vectorized(data, t , window_size) for t in range(data_samples)] # Usa la función vectorizada

        # Crea las estadísticas para guardar
    trader.profit_history = []
    trader.epsilon_history = []
    trader.trades_history = []
    trader.loss_history = []
    trader.drawdown_history = []
    trader.sharpe_ratios = []
    trader.accuracy_history = []
    trader.avg_win_history = []
    trader.avg_loss_history = []
    trader.epsilon = 1.0
    trader.step_counter = 0

    state = states[0]
    total_profit_pips = 0
    trader.inventory = []
    trades_count = 0
    wins = 0
    losses = 0
    worse_equity = 9999999
    profit_dollars = 0
    profit_dollars_total = 0
    winning_profits_pips = []
    losing_profits_pips = []
    peak_equity = balance_first
    current_equity = balance_first
    current_drawdown_real =balance_first
    peak_equity_drawdrown_real = balance_first
    drawdown_history_episode = []
    drawdown_real_history_episode = []
    episode_returns_pips = []

            
    best_low =9999999
            
    trader.total_rewards = 0
            

            # Bucle que recorre cada estado en los datos que descargue de mt5
    for t in range(data_samples):
 
                # La ia toma una decision
        action = trader.trade(state)
            
                # Precio actual
        current_price = data['close'].iloc[t].item()
        current_low = data['low'].iloc[t].item()
        spread = data['spread'].iloc[t].item() * lot_size
                # Indice del precio en el estado actual
        timestamp = data.index[t]

                # Coloca el precio de compra actual con
        buy_price = current_price +  spread / 2 if action == 1 else current_price
        sell_price = current_price - spread / 2 if action == 2 and len(trader.inventory) > 0 else current_price
                
                
        if len(trader.inventory) > 0 and best_low > current_low:
            best_low = current_low
        elif len(trader.inventory) <= 0:
            best_low = 9999999
           
                # Si la accion de la ia es igua a 1 compra
        if action == 1 and not trader.inventory :  # Comprar
                    # Agrega el precio de compra al inventario
            trader.inventory.append(buy_price)
                    
                    
                # Si la accion de la ia es igual a 2 y hay un trade abierto vende esa accion
        elif action == 2 and len(trader.inventory) > 0:  # Vender
                    # Toma el precio que compro el activo
            original_buy_price = trader.inventory.pop(0)
                    # Calcula el profit que obtuvo de la venta de activo (en pips)
            if(es_indice):
                profit_pips = (sell_price - original_buy_price) 
                ticks = profit_pips / 0.25   
                profit_dollars = ticks * tick_value * (lot_size / 1.0)
            elif(es_forex):
                profit_pips = (sell_price - original_buy_price) * pip_multiplier
                        # Calcula la ganancia/pérdida en dólares
                profit_dollars = profit_pips * pip_value_eur_usd
            elif(es_metal):
                profit_pips = (sell_price - original_buy_price) 
                pip_drawdrow_real = (  best_low - original_buy_price)
                        # Calcula la ganancia/pérdida en dólares
                profit_drawdrow_real = (pip_drawdrow_real * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size) 
                profit_dollars = (profit_pips * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size) 
                        
                    # Coloco el profit en la variable (en pips)
            total_profit_pips += profit_pips
                    # Usando la variable pip_value que ya definiste
            profit_dollars_total += profit_dollars
                    
                    # Suma al contador de trades
            trades_count += 1
       
            current_drawdown_real +=profit_drawdrow_real
            current_equity += profit_dollars  
                    
            if current_equity > peak_equity:
                peak_equity = current_equity
            elif current_equity < worse_equity:
                worse_equity = current_equity
                    
            if current_drawdown_real > peak_equity_drawdrown_real:
                peak_equity_drawdrown_real = current_drawdown_real

                    # Agrega el retorno al retorno del episodio cada profit (en pips)
            episode_returns_pips.append(profit_pips)
     
                    # Verifica si el profit salio ganador agrefa uno a wins y agrega el profit a winning_profits (en pips)
            if profit_pips > 0:
                wins += 1
                winning_profits_pips.append(profit_pips)
                    # De lo contrario si sale perdedor se agrega uno a lose y agrega el profit a losing_profits (en pips)
            else:
                losses += 1
                losing_profits_pips.append(profit_pips)

            
            print(f"tiempo {t} de {data_samples}  , por eleccion de ia  Venta a {sell_price:.5f}, Compra a {original_buy_price:.5f}, Profit (pips): {profit_pips:.2f}, Profit (USD): {profit_dollars:.2f}, total de dinero actual {current_equity:.2f}")
            print("")
            print(f"suma de recompensa {trader.total_rewards}")
            print("")
                    
        elif action == 3 and len(trader.inventory_sell) <= 0:   
            trader.inventory_sell.append(sell_price)
                                       
                
        elif action == 4 and len(trader.inventory_sell) > 0:
            original_sell_price = trader.inventory_sell.pop(0)
                    # Calcula el profit que obtuvo de la venta de activo (en pips)
            if(es_indice):
                profit_pips = (original_sell_price - buy_price  ) 
                ticks = profit_pips / 0.25   
                profit_dollars = ticks * tick_value * (lot_size / 1.0)
            elif(es_forex):
                profit_pips = (original_sell_price - buy_price ) * pip_multiplier
                       # Calcula la ganancia/pérdida en dólares
                profit_dollars = profit_pips * pip_value_eur_usd
            elif(es_metal):
                profit_pips = (original_sell_price - buy_price ) 
                pip_drawdrow_real = (  best_low - original_sell_price)
                        # Calcula la ganancia/pérdida en dólares
                profit_drawdrow_real = (pip_drawdrow_real * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size) 
                profit_dollars = (profit_pips * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size) 
                        
                    # Coloco el profit en la variable (en pips)
            total_profit_pips += profit_pips
                    # Usando la variable pip_value que ya definiste
            profit_dollars_total += profit_dollars
                    
                    # Suma al contador de trades
            trades_count += 1
       
            current_drawdown_real +=profit_drawdrow_real
            current_equity += profit_dollars  
                    
            if current_equity > peak_equity:
                peak_equity = current_equity
            elif current_equity < worse_equity:
                worse_equity = current_equity
                    
            if current_drawdown_real > peak_equity_drawdrown_real:
                peak_equity_drawdrown_real = current_drawdown_real

                    # Agrega el retorno al retorno del episodio cada profit (en pips)
            episode_returns_pips.append(profit_pips)
                    
     
                    # Verifica si el profit salio ganador agrefa uno a wins y agrega el profit a winning_profits (en pips)
            if profit_pips > 0:
                wins += 1
                winning_profits_pips.append(profit_pips)
                    # De lo contrario si sale perdedor se agrega uno a lose y agrega el profit a losing_profits (en pips)
            else:
                losses += 1
                losing_profits_pips.append(profit_pips)

                        
            
            print(f"tiempo {t} de {data_samples} ,por eleccion de ia CERRO Venta a {original_sell_price:.5f}, Compra a {buy_price:.5f}, Profit (pips): {profit_pips:.2f}, Profit (USD): {profit_dollars:.2f}, total de dinero actual {current_equity:.2f}")
            print("")
            print(f"suma de recompensa {trader.total_rewards}")
            print("")
                    
                    
        elif len(trader.inventory) > 0 and 'time' in data.columns and data['time'].notnull().all() and int(hora.iloc[t].split(":")[0]) == 23   :
                    # Toma el precio que compro el activo
            original_buy_price = trader.inventory.pop(0)
                    # Calcula el profit que obtuvo de la venta de activo (en pips)
            if(es_indice):
                profit_pips = (sell_price - original_buy_price) 
                ticks = profit_pips / 0.25   
                profit_dollars = ticks * tick_value * (lot_size / 1.0)
            elif(es_forex):
                profit_pips = (sell_price - original_buy_price) * pip_multiplier
                        # Calcula la ganancia/pérdida en dólares
                profit_dollars = profit_pips * pip_value_eur_usd
            elif(es_metal):
                profit_pips = (sell_price - original_buy_price) 
                        # Calcula la ganancia/pérdida en dólares
                pip_drawdrow_real = (  best_low - original_buy_price)
                profit_drawdrow_real = (pip_drawdrow_real * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size)
                profit_dollars =( profit_pips * pip_value_eur_usd) - ( trader.commission_per_trade * lot_size) 
                    
                    
                    # vender forzadamente por cierre de jornada, por ejemplo      
                    # Coloco el profit en la variable (en pips)
            total_profit_pips += profit_pips
                    # Usando la variable pip_value que ya definiste
            profit_dollars_total +=  profit_dollars
                    

                    # Suma al contador de trades
            trades_count += 1
                    
            current_drawdown_real +=profit_drawdrow_real
            current_equity += profit_dollars # Asumiendo EUR es la base
                    
            if current_equity > peak_equity:
                peak_equity = current_equity
            elif current_equity < worse_equity:
                worse_equity = current_equity
                    
            if current_drawdown_real > peak_equity_drawdrown_real:
                peak_equity_drawdrown_real = current_drawdown_real

                    # Agrega el retorno al retorno del episodio cada profit (en pips)
            episode_returns_pips.append(profit_pips)

                    # Verifica si el profit salio ganador agrefa uno a wins y agrega el profit a winning_profits (en pips)
            if profit_pips > 0:
                wins += 1
                winning_profits_pips.append(profit_pips)
                    # De lo contrario si sale perdedor se agrega uno a lose y agrega el profit a losing_profits (en pips)
            else:
                losses += 1
                losing_profits_pips.append(profit_pips)

                    # Puedes imprimir o guardar la ganancia en dólares si lo deseas
                   #print("")
                   
            print(f"tiempo {t} de {data_samples} , Venta a {sell_price:.5f}, Compra a {original_buy_price:.5f}, Profit (pips): {profit_pips:.2f}, Profit (USD): {profit_dollars:.2f}, total de dinero actual {current_equity:.2f}")
    
                             
                                    
        drawdown_real = (peak_equity_drawdrown_real - current_drawdown_real) / peak_equity_drawdrown_real if peak_equity_drawdrown_real != 0 else 0
                # Calculo de drawdown como ($1070 - $1050) / $1070 ≈ 0.0187 o 1.87%.
        drawdown = (peak_equity - current_equity) / peak_equity if peak_equity != 0 else 0
                # Se agrega al array para ver cual es el max drawdown 
                
              
        drawdown_history_episode.append(drawdown)
        drawdown_real_history_episode.append(drawdown_real)


        profit_dollars = 0


    #sharpe = calculate_sharpe_ratio(np.array(episode_returns_pips))
    accuracy = wins / trades_count if trades_count > 0 else 0
    avg_win = np.mean(winning_profits_pips) if winning_profits_pips else 0
    avg_loss = np.mean(losing_profits_pips) if losing_profits_pips else 0
    max_drawdown = max(drawdown_history_episode) if drawdown_history_episode else 0
    max_drawdown_real = max(drawdown_real_history_episode) if drawdown_real_history_episode else 0
            #if(episode % 5 == 0):
               # enviar_alerta(f"Jefe vamos en el episodio {episode} y el total de recompensa es de {trader.total_rewards:.2f}. con un total en dinero de: {price_format(profit_dollars_total)} ")

   
    print("")
    print("===========================================================================")
    print(f"""Fin 
                  Beneficio (pips)={total_profit_pips}
                  Beneficio (usd)={price_format(profit_dollars_total)}
                  Trades={trades_count}
                  Peor balance={worse_equity}
                  Mejor Balance = {peak_equity}
                  wins={wins}
                  Maximo drawdown = {max_drawdown_real:.2%}
                  Drawdown={max_drawdown:.2%}
                  Accuracy={accuracy:.2%}
                  """)
    print("")
    print(f"total de recompensas: {trader.total_rewards:.2f} ")
    print("===========================================================================")
    print("")






