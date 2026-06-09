import matplotlib.pyplot as plt
import pandas as pd

# ===================== ДАННЫЕ =====================
performance_data = {
    'task_name': [
        "ken-07.mps", "ken-11.mps", "osa-07.mps", "osa-14.mps",
        "osa-30.mps", "osa-60.mps", "pds-02.mps",
        "fit2d.mps", "fit2p.mps", "woodw.mps"
    ],
    'pami_time': [2900, 68000, 4700, 15000, 55000, 310000, 1010, 2619, 46000, 1400],
    'pami_major_iter': [304, 1898, 489, 928, 2124, 3655, 144, 86, 4663, 398],
    'pami_avg_mini': [3.68, 3.707, 2.109, 2.11, 1.88, 2.11, 3.45, 2.2907, 2.12, 2.7],
    'cuda_time': [3300, 44764, 3381, 14200, 34481, 134000, 1900, 1597, 43000, 2400],
    'cuda_iter': [1078, 6845, 683, 1558, 3783, 7479, 833, 128, 6527, 981],
    'seq_time': [2100, 71000, 6687, 27000, 92688, 507000, 1564, 2900, 45000, 3200],
    'seq_iter': [1079, 6917, 754, 1673, 3540, 8484, 817, 134, 6834, 833]
}
df_perf = pd.DataFrame(performance_data)
df_perf['task_base_name'] = df_perf['task_name'].str.replace('.mps', '').str.replace('-a', '').str.replace('-b', '')

# ============ РАЗМЕРНОСТИ (только столбцы) ============
netlib_info = {
    'ken-07': {'cols': 3602},
    'ken-11': {'cols': 21349},
    'osa-07': {'cols': 25067},
    'osa-14': {'cols': 54797},
    'osa-30': {'cols': 104374},
    'osa-60': {'cols': 243246},
    'pds-02': {'cols': 7716},
    'fit2d': {'cols': 10500},
    'woodw': {'cols': 8418},
}

for base_name, info in netlib_info.items():
    mask = df_perf['task_base_name'] == base_name
    df_perf.loc[mask, 'num_cols'] = info['cols']

# Расчёт общего числа итераций для CPU (мажор * среднее мини)
df_perf['pami_total_iter'] = df_perf['pami_major_iter'] * df_perf['pami_avg_mini']
# Время на одну (мини-)итерацию для CPU
df_perf['pami_time_per_iter'] = df_perf['pami_time'] / df_perf['pami_total_iter']

# Для GPU: общее число итераций уже дано как cuda_iter
df_perf['cuda_time_per_iter'] = df_perf['cuda_time'] / df_perf['cuda_iter']
df_perf['seq_time_per_iter'] = df_perf['seq_time'] / df_perf['seq_iter']

# Ускорение (по общему времени)
df_perf['speedup_pami'] = df_perf['seq_time'] / df_perf['pami_time']
df_perf['speedup_cuda'] = df_perf['seq_time'] / df_perf['cuda_time']

# Общее время в секундах
df_perf['pami_time_sec'] = df_perf['pami_time'] / 1000.0
df_perf['cuda_time_sec'] = df_perf['cuda_time'] / 1000.0
df_perf['seq_time_sec'] = df_perf['seq_time'] / 1000.0

# Сортировка по числу столбцов
df = df_perf.sort_values('num_cols')

# ===================== СТИЛИ =====================
plt.style.use('seaborn-v0_8-whitegrid')
colors = {'cpu': '#2E86AB', 'cuda': '#A23B72', 'seq': '#F18F01'}

def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', labelsize=10)

def annotate_points(ax, x, y, labels, offset=(5, 5), fontsize=8, alpha=0.7):
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=offset, textcoords='offset points',
                    fontsize=fontsize, alpha=alpha)

# ========== 1. CPU параллельный ==========
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

# Speedup
ax = axes1[0]
ax.semilogx(df['num_cols'], df['speedup_pami'], 'o-', color=colors['cpu'], linewidth=2, markersize=8, label='CPU параллельный / Последовательный')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
style_ax(ax, 'Ускорение CPU параллельный / Последовательный', 'Число столбцов', 'Ускорение (раз)')
annotate_points(ax, df['num_cols'], df['speedup_pami'], df['task_base_name'])
ax.legend()

# Time per iteration (мини-итерации для CPU)
ax = axes1[1]
ax.loglog(df['num_cols'], df['seq_time_per_iter'], 's-', color=colors['seq'], linewidth=2, markersize=8, label='Последовательный')
ax.loglog(df['num_cols'], df['pami_time_per_iter'], 'o-', color=colors['cpu'], linewidth=2, markersize=8, label='CPU параллельный')
style_ax(ax, 'Время на итерацию', 'Число столбцов', 'Время (мс / итер)')
ax.legend()

# Total time (seconds)
ax = axes1[2]
ax.loglog(df['num_cols'], df['seq_time_sec'], 's-', color=colors['seq'], linewidth=2, markersize=8, label='Последовательный')
ax.loglog(df['num_cols'], df['pami_time_sec'], 'o-', color=colors['cpu'], linewidth=2, markersize=8, label='CPU параллельный')
style_ax(ax, 'Общее время', 'Число столбцов', 'Время (с)')
ax.legend()

plt.tight_layout()
plt.savefig('cpu_plots.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

# ========== 2. GPU параллельный ==========
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# Speedup
ax = axes2[0]
ax.semilogx(df['num_cols'], df['speedup_cuda'], 's-', color=colors['cuda'], linewidth=2, markersize=8, label='GPU параллельный / Последовательный')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
style_ax(ax, 'Ускорение GPU параллельный / Последовательный', 'Число столбцов', 'Ускорение (раз)')
annotate_points(ax, df['num_cols'], df['speedup_cuda'], df['task_base_name'])
ax.legend()

# Time per iteration
ax = axes2[1]
ax.loglog(df['num_cols'], df['seq_time_per_iter'], 's-', color=colors['seq'], linewidth=2, markersize=8, label='Последовательный')
ax.loglog(df['num_cols'], df['cuda_time_per_iter'], 'o-', color=colors['cuda'], linewidth=2, markersize=8, label='GPU параллельный')
style_ax(ax, 'Время на итерацию', 'Число столбцов', 'Время (мс / итер)')
ax.legend()

# Total time (seconds)
ax = axes2[2]
ax.loglog(df['num_cols'], df['seq_time_sec'], 's-', color=colors['seq'], linewidth=2, markersize=8, label='Последовательный')
ax.loglog(df['num_cols'], df['cuda_time_sec'], 'o-', color=colors['cuda'], linewidth=2, markersize=8, label='GPU параллельный')
style_ax(ax, 'Общее время', 'Число столбцов', 'Время (с)')
ax.legend()

plt.tight_layout()
plt.savefig('gpu_plots.png', dpi=300, bbox_inches='tight')
plt.close(fig2)

# ========== 3. СОВМЕЩЁННЫЙ (без подписей точек) ==========
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

# Speedup
ax = axes3[0]
ax.semilogx(df['num_cols'], df['speedup_pami'], 'o-', color=colors['cpu'], linewidth=2, markersize=8, label='CPU параллельный')
ax.semilogx(df['num_cols'], df['speedup_cuda'], 's-', color=colors['cuda'], linewidth=2, markersize=8, label='GPU параллельный')
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
style_ax(ax, 'Ускорение относительно последовательного', 'Число столбцов', 'Ускорение (раз)')
ax.legend()

# Time per iteration
ax = axes3[1]
ax.loglog(df['num_cols'], df['pami_time_per_iter'], 'o-', color=colors['cpu'], linewidth=2, markersize=8, label='CPU параллельный')
ax.loglog(df['num_cols'], df['cuda_time_per_iter'], 's-', color=colors['cuda'], linewidth=2, markersize=8, label='GPU параллельный')
style_ax(ax, 'Время на одну итерацию', 'Число столбцов', 'Время (мс / итер)')
ax.legend()

# Total time (seconds)
ax = axes3[2]
ax.loglog(df['num_cols'], df['pami_time_sec'], 'o-', color=colors['cpu'], linewidth=2, markersize=8, label='CPU параллельный')
ax.loglog(df['num_cols'], df['cuda_time_sec'], 's-', color=colors['cuda'], linewidth=2, markersize=8, label='GPU параллельный')
style_ax(ax, 'Общее время выполнения', 'Число столбцов', 'Время (с)')
ax.legend()

plt.tight_layout()
plt.savefig('combined_plots.png', dpi=300, bbox_inches='tight')
plt.close(fig3)

print("✅ Готово:")
print("   - cpu_plots.png    (CPU параллельный, время на мини-итерацию)")
print("   - gpu_plots.png    (GPU параллельный, общее число итераций из таблицы)")
print("   - combined_plots.png (сравнение, подписей точек нет)")