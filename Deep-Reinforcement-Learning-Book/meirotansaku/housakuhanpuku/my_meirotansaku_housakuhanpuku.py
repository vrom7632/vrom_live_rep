import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def init_byouga(plt):
    """ 迷路の初期描画"""
    # 迷路を描画
    fig = plt.figure(figsize=(5, 5))
    
    ax = plt.gca()

    # 壁の描画
    plt.plot([1,1], [0,1], color='red', linewidth=2)
    plt.plot([1,2], [2,2], color='red', linewidth=2)
    plt.plot([2,2], [2,1], color='red', linewidth=2)
    plt.plot([2,3], [1,1], color='red', linewidth=2)

    # 状態を示す文字の描画
    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')
    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')
    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')
    plt.text(0.5, 2.3, 'START', ha='center')
    plt.text(2.5, 0.3, 'GOAL', ha='center')

    # 描画範囲の設定
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # 現在地S0に緑丸を描画する
    line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)
    return fig,line

def init_theta(np):
    """ thetaの初期値の決定 """
    # 行は状態0～7、列は移動方向で↑、→、↓、←を表す
    return np.array([[np.nan, 1, 1, np.nan],  # snp.nan
                        [np.nan, 1, np.nan, 1],  # s1
                        [np.nan, np.nan, 1, 1],  # s2
                        [1, 1, 1, np.nan],  # s3
                        [np.nan, np.nan, 1, 1],  # s4
                        [1, np.nan, np.nan, np.nan],  # s5
                        [1, np.nan, np.nan, np.nan],  # s6
                        [1, 1, np.nan, np.nan],  # s7、※s8はゴールなので、方策はなし
                        ],dtype=float)

def softmax_convert_into_pi_from_theta(theta):
    """ソフトマックス関数で割合を計算する                """
    """theta:numpy array float                        """
    theta = np.exp(theta)  # thetaをexp(theta)へと変換 ※教材では1.0 * thenta
    theta = np.array(list(map(lambda th:th/np.nansum(th),theta))) #割合の計算
     
    return np.nan_to_num(theta)

def get_action_and_next_s(pi, s):
    """行動aと1step移動後の状態sを求める関数を定義"""
    up = -3
    right = 1
    down = 3
    left = -1
    direction = [up, right, down, left]
    action = [0, 1, 2, 3]
    next = np.random.choice(list(range(0,len(direction))), p=pi[s])

    return [action[next], s + direction[next]]

def goal_maze_ret_s_a(pi):
    """迷路を解く関数の定義、状態と行動の履歴を出力"""
    s = 0  # スタート地点
    s_a_history = [[0, np.nan]]  # エージェントの移動を記録するリスト

    while True:  # ゴールするまでループ
        [action, next_s] = get_action_and_next_s(pi, s)
        s_a_history[-1][1] = action
        # 現在の状態（つまり一番最後なのでindex=-1）の行動を代入

        s_a_history.append([next_s, np.nan])
        # 次の状態を代入。行動はまだ分からないのでnanにしておく

        if next_s == 8:  # ゴール地点なら終了
            break
        else:
            s = next_s

    return s_a_history

def update_theta(theta, pi, s_a_history):
    """thetaの更新関数を定義します"""
    eta = 0.5 # 学習率
    T = len(s_a_history) - 1  # ゴールまでの総ステップ数

    [m, n] = theta.shape  # thetaの行列サイズを取得
    delta_theta = theta.copy()  # Δthetaの元を作成、ポインタ参照なので、delta_theta = thetaはダメ

    # delta_thetaを要素ごとに求めます
    for i in range(0, m):
        for j in range(0, n):
            if not(np.isnan(theta[i, j])):  # thetaがnanでない場合

                SA_i = [SA for SA in s_a_history if SA[0] == i]
                # 履歴から状態iのものを取り出すリスト内包表記です

                SA_ij = [SA for SA in s_a_history if SA == [i, j]]
                # 状態iで行動jをしたものを取り出す

                N_i = len(SA_i)  # 状態iで行動した総回数
                N_ij = len(SA_ij)  # 状態iで行動jをとった回数
                delta_theta[i, j] = (N_ij - pi[i, j] * N_i) / T

    return theta + eta * delta_theta

def agent_animation(s_a_history,fig,line):
    """初期化関数とフレームごとの描画関数を用いて動画を作成する"""
    def init_background():
        """背景画像の初期化"""
        line.set_data([], [])
        return (line,)

    def animate(i):
        """フレームごとの描画内容"""
        state = s_a_history[i][0]  # 現在の場所を描く
        x = (state % 3) + 0.5  # 状態のx座標は、3で割った余り+0.5
        y = 2.5 - int(state / 3)  # y座標は3で割った商を2.5から引く
        line.set_data(x, y)
        return (line,)

    return animation.FuncAnimation(fig, animate, init_func=init_background, frames=len(
        s_a_history), interval=400, repeat=False)

if __name__ == "__main__":
    # 迷路の初期描画
    fig,line = init_byouga(plt)

    # 探索可能位置の初期化
    theta_0 = init_theta(np)
    
    # 初期の方策pi_0を求める
    pi_0 = softmax_convert_into_pi_from_theta(theta_0)

    # 初期の方策で迷路を解く
    s_a_history = goal_maze_ret_s_a(pi_0)
    print(s_a_history)
    print("step:{}".format(str(len(s_a_history) - 1)))
    ani = agent_animation(s_a_history, fig, line)
    ani.save("meirotansaku_housaku_1_gakushumae.mp4", writer="ffmpeg")

    # 方策勾配法で迷路を解く
    stop_epsilon = 10**-4  # 10^-4よりも方策に変化が少なくなったら学習終了とする

    theta = theta_0
    pi = pi_0
    count = 0
    while True:
        s_a_history = goal_maze_ret_s_a(pi)  # 方策πで迷路内を探索した履歴を求める
        new_theta = update_theta(theta, pi, s_a_history)  # パラメータΘを更新
        new_pi = softmax_convert_into_pi_from_theta(new_theta)  # 方策πの更新
        print(pi)
        print("epsilon:{}".format(np.sum(np.abs(new_pi - pi))))  # 方策の変化を出力
        print("step:{}".format(str(len(s_a_history) - 1)))
        
        if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
            break
        else:
            theta = new_theta
            pi = new_pi
            count += 1
            # 学習経過のアニメーション描画
            if count % 100 == 0: 
                ani = agent_animation(s_a_history, fig, line)
                ani.save("meirotansaku_housaku_2_gakushuchu_{}.mp4".format(count), writer="ffmpeg")
            
    print("count:{}".format(count))
    # 迷路探索のアニメーション描画
    ani = agent_animation(s_a_history, fig, line)
    ani.save("meirotansaku_housaku_3_gakushugo.mp4", writer="ffmpeg")
