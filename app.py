import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import silhouette_score
from scipy.stats import linregress
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

# 设置中文字体和深色主题
# 更完善的中文字体配置
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',      # 微软雅黑 (Windows首选)
    'SimHei',               # 黑体 (Windows备选)
    'SimSun',               # 宋体 (Windows备选)
    'PingFang SC',          # 苹方 (macOS)
    'Hiragino Sans GB',     # 冬青黑体 (macOS)
    'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux)
    'DejaVu Sans',          # 通用备选
    'Arial Unicode MS',     # Unicode支持
    'Noto Sans CJK SC'      # 思源黑体 (Google)
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体族

# 先设置基础样式
plt.style.use('dark_background')

# 然后自定义深色主题颜色 - 使用十六进制颜色
plt.rcParams['figure.facecolor'] = '#0f172a'  # 深色背景
plt.rcParams['axes.facecolor'] = '#1e293b'
plt.rcParams['axes.edgecolor'] = '#e2e8f0'
plt.rcParams['axes.labelcolor'] = '#e2e8f0'
plt.rcParams['xtick.color'] = '#e2e8f0'
plt.rcParams['ytick.color'] = '#e2e8f0'
plt.rcParams['grid.color'] = '#475569'  # 使用十六进制颜色
plt.rcParams['text.color'] = '#e2e8f0'
plt.rcParams['legend.facecolor'] = '#1e293b'
plt.rcParams['legend.edgecolor'] = '#475569'
plt.rcParams['legend.labelcolor'] = '#e2e8f0'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# 提高图表质量和分辨率设置
plt.rcParams['figure.dpi'] = 150  # 提高DPI
plt.rcParams['savefig.dpi'] = 150  # 保存时的高DPI
plt.rcParams['savefig.format'] = 'png'  # 使用PNG格式
plt.rcParams['savefig.bbox'] = 'tight'  # 紧凑边界
plt.rcParams['savefig.pad_inches'] = 0.1  # 减少边距

# 抗锯齿设置
plt.rcParams['lines.antialiased'] = True
plt.rcParams['patch.antialiased'] = True
plt.rcParams['text.antialiased'] = True

# 字体渲染优化
# 移除冲突的字体设置，使用上面配置的中文字体优先级
plt.rcParams['text.hinting'] = 'auto'
plt.rcParams['text.hinting_factor'] = 8

# 数据大屏专用颜色
DASHBOARD_PLOT_COLORS = [
    '#00d4ff',  # 青色 - 主色
    '#7c3aed',  # 紫色 - 次色
    '#ff006e',  # 粉色 - 强调色
    '#10b981',  # 绿色 - 成功
    '#f59e0b',  # 橙色 - 警告
    '#06b6d4',  # 天蓝 - 信息
    '#f97316',  # 深橙
    '#8b5cf6',  # 浅紫
    '#ec4899',  # 粉红
    '#14b8a6'   # 青绿
]

# ============================================================================
# 页面配置
# ============================================================================
st.set_page_config(
    page_title="电商销售分析与策略优化系统",
    page_icon="📈",  # 这个无法使用Bootstrap Icons，只能用emoji
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 现代化简洁UI样式
# ============================================================================
st.markdown("""
<link rel="stylesheet" href="./node_modules/bootstrap-icons/font/bootstrap-icons.css">

<style>
    /* 数据大屏样式 */
    .stApp {
        background: #0a0e1a;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        overflow-x: hidden;
    }

    /* 隐藏侧边栏 - 使用顶部导航 */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* 通用文字颜色 - 确保在深色主题下可见 */
    * {
        color: #e2e8f0 !important;
    }

    /* 强制覆盖所有数据表格文字 */
    .stDataFrame,
    .stDataFrame *,
    .dvn-scroller,
    .dvn-scroller *,
    .stDataFrameGlideDataEditor,
    .stDataFrameGlideDataEditor *,
    [data-testid="stDataFrame"] {
        color: #ffffff !important;
    }

    /* 表格单元格和表头 */
    td, th, .cell, .header {
        color: #ffffff !important;
    }

    /* 特殊元素颜色覆盖 */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #ffffff !important;
    }

    .stText {
        color: #e2e8f0 !important;
    }

    .stCaption {
        color: #cbd5e1 !important;
    }

    p, span, div, label {
        color: #e2e8f0 !important;
    }

    .metric-value {
        color: #ffffff !important;
    }

    /* 主内容区域 - 全屏大屏 */
    .main .block-container {
        padding-top: 20px;
        background: transparent;
        max-width: 100%;
        padding-left: 20px;
        padding-right: 20px;
    }

    /* 大屏标题 */
    .dashboard-header {
        text-align: center;
        margin-bottom: 40px;
        position: relative;
    }

    .dashboard-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(45deg, #00d4ff, #7c3aed, #ff006e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-transform: uppercase;
        letter-spacing: 4px;
        margin-bottom: 15px;
        text-shadow: 0 0 40px rgba(0,212,255,0.6);
        animation: glow 3s ease-in-out infinite alternate;
    }

    .dashboard-subtitle {
        font-size: 1.3rem;
        color: #e2e8f0;
        letter-spacing: 2px;
        font-weight: 300;
    }

    /* 数据卡片网格 */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 25px;
        margin-bottom: 40px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 20px;
        padding: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        transition: all 0.4s ease;
        backdrop-filter: blur(10px);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #ff006e);
        animation: glow 2s ease-in-out infinite alternate;
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 60px rgba(0,212,255,0.4);
        border-color: #00d4ff;
    }

    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 15px;
        text-shadow: 0 0 20px rgba(255,255,255,0.3);
    }

    .metric-label {
        font-size: 1rem;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 20px;
        font-weight: 500;
    }

    .metric-change {
        font-size: 0.9rem;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 8px;
        display: inline-block;
    }

    .metric-positive {
        background: rgba(16,185,129,0.2);
        color: #10b981;
        border: 1px solid rgba(16,185,129,0.4);
    }

    .metric-negative {
        background: rgba(239,68,68,0.2);
        color: #ef4444;
        border: 1px solid rgba(239,68,68,0.4);
    }

    /* 图表容器 */
    .chart-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }

    .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
    }

    .chart-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .chart-icon {
        width: 30px;
        height: 30px;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 16px;
    }

    /* 实时数据指示器 */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(16,185,129,0.2);
        color: #10b981;
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid rgba(16,185,129,0.4);
        margin-left: auto;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .live-dot {
        width: 10px;
        height: 10px;
        background: #10b981;
        border-radius: 50%;
        animation: pulse 2s infinite;
        box-shadow: 0 0 10px rgba(16,185,129,0.8);
    }

    /* 状态面板 */
    .status-panel {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
    }

    .status-item {
        text-align: center;
        flex: 1;
        min-width: 140px;
        padding: 20px;
        background: rgba(255,255,255,0.03);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }

    .status-item:hover {
        background: rgba(255,255,255,0.06);
        transform: translateY(-3px);
    }

    .status-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 8px;
        text-shadow: 0 0 15px rgba(255,255,255,0.3);
    }

    .status-label {
        font-size: 0.9rem;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
    }

    /* 导航标签样式 */
    .nav-tabs {
        display: flex;
        gap: 12px;
        margin-bottom: 40px;
        background: rgba(30,41,59,0.6);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(71,85,105,0.6);
    }

    .nav-tab {
        padding: 15px 30px;
        background: rgba(51,65,85,0.8);
        border: 1px solid #475569;
        border-radius: 12px;
        color: #e2e8f0;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: pointer;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }

    .nav-tab:hover {
        background: rgba(71,85,105,0.8);
        color: #ffffff;
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,212,255,0.2);
    }

    .nav-tab.active {
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        color: #ffffff;
        border-color: #00d4ff;
        box-shadow: 0 8px 30px rgba(0,212,255,0.4);
        transform: translateY(-3px);
    }

    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 30px;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0,212,255,0.3);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-size: 0.9rem;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #00b8e6, #6d28d9);
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,212,255,0.5);
    }

    /* 输入框样式 */
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(30,41,59,0.8);
        border: 1px solid #475569;
        border-radius: 10px;
        padding: 12px 18px;
        color: #ffffff;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }

    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 0 4px rgba(0,212,255,0.15);
    }

    /* 信息框样式 */
    .info-box {
        background: linear-gradient(135deg, rgba(0,212,255,0.15) 0%, rgba(124,58,237,0.15) 100%);
        color: #e0f2fe;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 1px solid rgba(0,212,255,0.3);
        box-shadow: 0 10px 30px rgba(0,212,255,0.2);
        backdrop-filter: blur(10px);
    }

    .success-box {
        background: linear-gradient(135deg, rgba(16,185,129,0.15) 0%, rgba(5,150,105,0.15) 100%);
        color: #d1fae5;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 1px solid rgba(16,185,129,0.3);
        box-shadow: 0 10px 30px rgba(16,185,129,0.2);
        backdrop-filter: blur(10px);
    }

    /* 任务状态卡片样式 */
    .task-status {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 15px;
        padding: 20px;
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        min-height: 100px;
    }

    .task-status:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }

    .status-completed {
        border-left: 4px solid #10b981;
        background: linear-gradient(135deg, rgba(16,185,129,0.1) 0%, #1e293b 100%);
    }

    .status-pending {
        border-left: 4px solid #f59e0b;
        background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, #1e293b 100%);
    }

    .status-completed .bi-check-circle-fill {
        color: #10b981;
    }

    .status-pending .bi-clock-fill {
        color: #f59e0b;
    }

    /* 数据表格样式 */
    .dataframe {
        background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(51,65,85,0.9) 100%);
        border: 1px solid #475569;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
    }

    /* 响应式设计 */
    @media (max-width: 768px) {
        .dashboard-title {
            font-size: 2.8rem;
            letter-spacing: 2px;
        }

        .metrics-grid {
            grid-template-columns: 1fr;
            gap: 20px;
        }

        .status-panel {
            flex-direction: column;
            gap: 15px;
        }

        .nav-tabs {
            flex-wrap: wrap;
        }

        .metric-value {
            font-size: 2.5rem;
        }

        .chart-container {
            padding: 20px;
        }
    }

    /* 动画效果 */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.4;
        }
    }

    @keyframes glow {
        0% {
            filter: brightness(1) drop-shadow(0 0 20px rgba(0,212,255,0.6));
        }
        100% {
            filter: brightness(1.2) drop-shadow(0 0 35px rgba(124,58,237,0.8));
        }
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(40px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .animate-slide-in {
        animation: slideIn 1s ease-out;
    }

    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #1a1f36;
        border-radius: 15px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        border-radius: 15px;
        border: 2px solid #1a1f36;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00b8e6, #6d28d9);
    }

    /* 文件上传组件深色主题样式 */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(51,65,85,0.9) 100%) !important;
        border: 2px dashed rgba(0,212,255,0.4) !important;
        border-radius: 16px !important;
        padding: 40px 20px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(0,212,255,0.1) !important;
        backdrop-filter: blur(10px) !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background: linear-gradient(135deg, rgba(30,41,59,0.95) 0%, rgba(51,65,85,0.95) 100%) !important;
        border-color: rgba(0,212,255,0.8) !important;
        box-shadow: 0 15px 40px rgba(0,212,255,0.3) !important;
        transform: translateY(-2px) !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #e2e8f0 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        text-align: center !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #e2e8f0 !important;
    }

    [data-testid="stFileUploaderDropzoneInstructions"] svg {
        color: #00d4ff !important;
        width: 48px !important;
        height: 48px !important;
        margin-bottom: 12px !important;
        filter: drop-shadow(0 0 10px rgba(0,212,255,0.6)) !important;
    }

    /* 文件上传组件拖拽时的效果 */
    [data-testid="stFileUploaderDropzone"][data-dragging="true"] {
        background: linear-gradient(135deg, rgba(0,212,255,0.15) 0%, rgba(124,58,237,0.15) 100%) !important;
        border-color: #00d4ff !important;
        box-shadow: 0 20px 50px rgba(0,212,255,0.4) !important;
        transform: scale(1.02) !important;
    }

    /* DataFrame表格深色主题样式 */
    .stDataFrameGlideDataEditor {
        background: linear-gradient(135deg, rgba(30,41,59,0.9) 0%, rgba(51,65,85,0.9) 100%) !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 25px rgba(0,212,255,0.15) !important;
        overflow: hidden !important;
        backdrop-filter: blur(10px) !important;
    }

    /* DataFrame滚动器 - 移除遮罩 */
    .dvn-scroller {
        background: transparent !important;
        border-radius: 12px !important;
        backdrop-filter: none !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* 移除表格组件的半透明背景 */
    .stDataFrameGlideDataEditor {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
    }

    /* 修复白色朦胧层 */
    .dvn-scroll-inner {
        background: transparent !important;
    }

    .dvn-stack {
        background: transparent !important;
    }

    .dvn-stack > div {
        background: transparent !important;
    }

    /* 恢复原始显示，不干扰Data Grid的正常功能 */

    /* 确保Data Grid容器正常 */
    .stDataFrame {
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
    }

    .stDataFrameGlideDataEditor:not(:has(.dvn-hidden)) {
        margin: 0 !important;
        width: 100% !important;
    }

    /* 确保表格内容可见 */
    .dvn-scroller .dvn-scroll-inner .dvn-stack {
        position: relative !important;
        z-index: 1 !important;
    }

    /* 移除可能的遮罩层 */
    .dvn-scroller::before,
    .dvn-scroller::after,
    .dvn-scroll-inner::before,
    .dvn-scroll-inner::after {
        display: none !important;
    }

    /* 数据表格文字颜色 - 更具体的选择器 */
    .dvn-scroller,
    .dvn-scroller *,
    .dvn-scroll-inner,
    .dvn-scroll-inner *,
    .dvn-stack,
    .dvn-stack * {
        color: #ffffff !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
    }

    /* 针对表格单元格 */
    .dvn-scroller td,
    .dvn-scroller th,
    .dvn-scroll-inner td,
    .dvn-scroll-inner th {
        color: #ffffff !important;
        font-weight: 500 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
    }

    /* 针对所有文本元素 */
    .stDataFrame [role="cell"],
    .stDataFrame [role="columnheader"],
    .stDataFrame .cell,
    .stDataFrame .header {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* Canvas表格样式 */
    [data-testid="data-grid-canvas"] {
        background: rgba(15, 23, 42, 0.95) !important;
        border-radius: 12px !important;
    }

    [data-testid="data-grid-canvas"] table {
        background: transparent !important;
    }

    [data-testid="data-grid-canvas"] th,
    [data-testid="data-grid-canvas"] td,
    [data-testid="data-grid-canvas"] [role="columnheader"],
    [data-testid="data-grid-canvas"] [role="gridcell"] {
        color: #ffffff !important;
        background: transparent !important;
        font-weight: 500 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
    }

    [data-testid="data-grid-canvas"] [role="columnheader"] {
        background: rgba(0,212,255,0.1) !important;
        color: #00d4ff !important;
        font-weight: 600 !important;
    }

    /* 针对具体的glide-cell元素 */
    [id^="glide-cell-"] {
        color: #ffffff !important;
        font-weight: 500 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5) !important;
    }

    /* 确保表格中的所有文本都是白色 */
    table[role="grid"] {
        color: #ffffff !important;
    }

    table[role="grid"] th,
    table[role="grid"] td,
    table[role="grid"] [role="columnheader"],
    table[role="grid"] [role="gridcell"] {
        color: #ffffff !important;
        background: transparent !important;
    }

    /* DataFrame表格内容区域 */
    .dvn-scroll-inner {
        background: transparent !important;
    }

    /* DataFrame表格头部 */
    .dvn-table-header {
        background: linear-gradient(135deg, rgba(0,212,255,0.2) 0%, rgba(124,58,237,0.2) 100%) !important;
        border-bottom: 2px solid rgba(0,212,255,0.4) !important;
    }

    .dvn-table-header th {
        background: transparent !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        border-right: 1px solid rgba(71,85,105,0.6) !important;
        font-size: 14px !important;
        text-align: center !important;
    }

    .dvn-table-header th:first-child {
        border-left: none !important;
    }

    .dvn-table-header th:last-child {
        border-right: none !important;
    }

    /* DataFrame表格单元格 */
    .dvn-table-cell {
        background: rgba(15,23,42,0.6) !important;
        color: #e2e8f0 !important;
        border-right: 1px solid rgba(71,85,105,0.4) !important;
        border-bottom: 1px solid rgba(71,85,105,0.4) !important;
        padding: 10px 14px !important;
        font-size: 13px !important;
        transition: all 0.2s ease !important;
    }

    .dvn-table-cell:hover {
        background: rgba(0,212,255,0.1) !important;
        color: #ffffff !important;
        box-shadow: inset 0 0 0 1px rgba(0,212,255,0.3) !important;
    }

    /* DataFrame表格行悬停效果 */
    .dvn-table-row:hover .dvn-table-cell {
        background: rgba(0,212,255,0.08) !important;
        color: #ffffff !important;
    }

    /* DataFrame表格选中行 */
    .dvn-table-row.selected .dvn-table-cell {
        background: rgba(0,212,255,0.15) !important;
        color: #ffffff !important;
        box-shadow: inset 0 0 0 2px rgba(0,212,255,0.4) !important;
    }

    /* DataFrame表格滚动条样式 */
    .dvn-scroller::-webkit-scrollbar {
        width: 8px !important;
        height: 8px !important;
    }

    .dvn-scroller::-webkit-scrollbar-track {
        background: rgba(30,41,59,0.4) !important;
        border-radius: 4px !important;
    }

    .dvn-scroller::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
        border-radius: 4px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }

    .dvn-scroller::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00b8e6, #6d28d9) !important;
    }

    /* DataFrame表格容器 */
    .stDataFrame {
        background: transparent !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* 隐藏Streamlit默认元素 */
    .stDeployButton {
        display: none;
    }

    /* 图表背景透明 */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }

    /* 全面图表深色主题样式 */
    /* Plotly图表 */
    .plotly {
        background: rgba(15,23,42,0.8) !important;
        border: 1px solid rgba(71,85,105,0.4) !important;
        border-radius: 12px !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
    }

    .js-plotly-plot {
        background: transparent !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* 图表SVG容器 */
    .plotly .svg-container {
        background: transparent !important;
        border-radius: 12px !important;
    }

    /* 图表标题 */
    .plotly .gtitle {
        fill: #ffffff !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }

    /* 图表坐标轴标签 */
    .plotly .xaxislayer-above text,
    .plotly .yaxislayer-above text {
        fill: #e2e8f0 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    /* 图表刻度标签 */
    .plotly .xtick text,
    .plotly .ytick text {
        fill: #cbd5e1 !important;
        font-size: 11px !important;
    }

    /* 图表网格线 */
    .plotly .gridlayer path {
        stroke: #475569 !important;
        stroke-width: 0.5 !important;
        opacity: 0.3 !important;
    }

    /* 图表图例 */
    .plotly .legend {
        background: rgba(30,41,59,0.9) !important;
        border: 1px solid rgba(71,85,105,0.6) !important;
        border-radius: 8px !important;
        backdrop-filter: blur(10px) !important;
    }

    .plotly .legend text {
        fill: #e2e8f0 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    /* 图表悬停提示 */
    .plotly .hoverlayer .hovertext {
        fill: #ffffff !important;
        background: rgba(15,23,42,0.95) !important;
        border: 1px solid rgba(0,212,255,0.4) !important;
        border-radius: 6px !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Matplotlib图表容器 */
    .stPlotlyChart,
    .stMatplotlibChart {
        background: rgba(15,23,42,0.8) !important;
        border: 1px solid rgba(71,85,105,0.4) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
        backdrop-filter: blur(10px) !important;
    }

    /* 强制所有图表容器背景 */
    .element-container:has(.stPlotlyChart),
    .element-container:has(.stMatplotlibChart) {
        background: transparent !important;
    }

    /* 图表图片 */
    .stPlotlyChart img,
    .stMatplotlibChart img {
        background: rgba(15,23,42,0.8) !important;
        border-radius: 8px !important;
    }

    /* Streamlit图表组件 */
    [data-testid="stPlotlyChart"],
    [data-testid="stMatplotlibChart"] {
        background: rgba(15,23,42,0.8) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 1px solid rgba(71,85,105,0.4) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3) !important;
    }

    /* 图表canvas元素 */
    canvas {
        background: rgba(15,23,42,0.8) !important;
        border-radius: 8px !important;
    }

    /* 通用图表修复 */
    svg {
        background: transparent !important;
    }

    /* 彻底去除所有图表白色背景和蒙版 */
    .streamlit-container .main .block-container {
        background-color: transparent !important;
        background: transparent !important;
    }

    /* 去除所有可能导致白色蒙版的元素 */
    .element-container,
    .stElement,
    [data-testid="element-container"],
    [data-testid="stElement"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }

    /* 专门针对图表的容器修复 */
    .element-container:has([data-testid="stPlotlyChart"]),
    .element-container:has([data-testid="stMatplotlibChart"]),
    [data-testid="stVerticalBlock"]:has([data-testid="stPlotlyChart"]),
    [data-testid="stVerticalBlock"]:has([data-testid="stMatplotlibChart"]) {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* 去除可能的半透明白色覆盖 */
    div:has(> .plotly),
    div:has(> [data-testid="stPlotlyChart"]),
    div:has(> [data-testid="stMatplotlibChart"]) {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* 强制去除backdrop-filter造成的模糊 */
    .plotly,
    .js-plotly-plot,
    .svg-container {
        backdrop-filter: none !important;
        filter: none !important;
    }

    /* 确保图表容器清晰 */
    canvas,
    svg {
        backdrop-filter: none !important;
        filter: none !important;
        -webkit-filter: none !important;
        opacity: 1 !important;
    }

    /* 去除Streamlit默认的卡片背景 */
    .st-emotion-cache-17v0vk3,
    .st-emotion-cache-zo524r {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* 去除任何可能的伪元素覆盖 */
    .element-container::before,
    .element-container::after,
    [data-testid="element-container"]::before,
    [data-testid="element-container"]::after {
        display: none !important;
    }

    /* 数据高亮效果 */
    .highlight {
        background: linear-gradient(135deg, rgba(0,212,255,0.4), rgba(124,58,237,0.4));
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 700;
        color: #ffffff;
    }

    /* 加载动画 */
    .loading-spinner {
        border: 4px solid rgba(0,212,255,0.2);
        border-top: 4px solid #00d4ff;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        box-shadow: 0 0 20px rgba(0,212,255,0.5);
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* 主标题调整 */
    h1 {
        color: #ffffff !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-bottom: 30px !important;
        text-shadow: 0 0 30px rgba(255,255,255,0.3) !important;
    }

    h2 {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin-bottom: 20px !important;
        text-shadow: 0 0 20px rgba(255,255,255,0.2) !important;
    }

    h3 {
        color: #ffffff !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        margin-bottom: 15px !important;
    }

    /* 最强制的图表修复 - 最高优先级 */
    body .streamlit-container .main .block-container .element-container,
    body [data-testid="element-container"] {
        background: transparent !important;
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* 强制修复图表容器 */
    body .streamlit-container [data-testid="stPlotlyChart"],
    body .streamlit-container [data-testid="stMatplotlibChart"],
    body .plotly,
    body .js-plotly-plot {
        background: rgba(15,23,42,0.9) !important;
        backdrop-filter: none !important;
        filter: none !important;
        -webkit-filter: none !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4) !important;
        border: 1px solid rgba(0,212,255,0.3) !important;
        border-radius: 12px !important;
    }

    /* 强制修复所有文字 */
    body .plotly text,
    body .js-plotly-plot text,
    body svg text {
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
    }

    /* 精确去除模糊效果 - 只禁用导致模糊的效果 */
    body {
        backdrop-filter: none !important;
    }

    .streamlit-container,
    .main,
    .block-container {
        backdrop-filter: none !important;
        filter: none !important;
    }

    /* 只禁用可能导致模糊的backdrop-filter */
    .element-container,
    [data-testid="element-container"] {
        backdrop-filter: none !important;
    }

    /* 图表容器禁用模糊效果 */
    .plotly,
    .js-plotly-plot,
    [data-testid="stPlotlyChart"],
    [data-testid="stMatplotlibChart"] {
        backdrop-filter: none !important;
    }

    /* 确保图表内容清晰 */
    .plotly svg,
    .js-plotly-plot svg {
        backdrop-filter: none !important;
        filter: none !important;
        transform: translateZ(0) !important; /* 强制硬件加速 */
    }

    /* 专门修复matplotlib图表图片 */
    .stMatplotlibChart img,
    .element-container:has(.stMatplotlibChart) img {
        background: #0f172a !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
        filter: none !important;
        image-rendering: -webkit-optimize-contrast !important;
        image-rendering: crisp-edges !important;
        image-rendering: pixelated !important;
    }

    /* 修复可能的缩放问题 */
    .stMatplotlibChart canvas,
    .stMatplotlibChart .canvas-container {
        background: #0f172a !important;
        image-rendering: -webkit-optimize-contrast !important;
        image-rendering: crisp-edges !important;
        transform: scale(1) !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State 初始化
# ============================================================================
def initialize_session_state():
    default_states = {
        'raw_data': None,  # 原始数据
        'step1_missing_data': None,  # 步骤1：缺失值统计结果
        'step2_price_data': None,  # 步骤2：进货价格处理后数据
        'step3_profit_data': None,  # 步骤3：利润修正后数据
        'step4_abnormal_data': None,  # 步骤4：异常修正及利润重算后数据
        'step5_minmax_data': None,  # 步骤5：MinMax标准化后数据
        'step5_zscore_data': None,  # 步骤5：ZScore标准化后数据
        'processed_data': None,  # 最终处理数据
        'category_encoder': None,  # 分类变量编码器
        'current_file': None,
        'task1_completed': False,
        'task2_completed': False,
        'task3_completed': False,
        'task4_completed': False,
        'column_types': None  # 字段类型
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================================
# 工具函数
# ============================================================================
def auto_detect_column_types(df):
    """自动识别字段类型：数值型、有序分类、无序分类、标识型"""
    column_types = {
        'numeric': [],  # 数值型（需标准化）
        'ordinal': [],  # 有序分类
        'nominal': [],  # 无序分类
        'identifier': []  # 标识型
    }

    # 标识型字段规则：唯一值占比>80% 或 字段名包含"ID/订单号/日期"
    id_keywords = ['id', '订单号', '日期', '编号', '序号']
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in id_keywords) or (df[col].nunique() / len(df) > 0.8):
            column_types['identifier'].append(col)
            continue

    # 数值型字段
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    column_types['numeric'] = [col for col in numeric_cols if col not in column_types['identifier']]

    # 分类字段（非数值、非标识）
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col not in column_types['identifier']]

    # 区分有序/无序分类
    ordinal_keywords = ['等级', '年龄', '评分', '段位', '层次']
    for col in categorical_cols:
        if any(keyword in col for keyword in ordinal_keywords):
            column_types['ordinal'].append(col)
        else:
            column_types['nominal'].append(col)

    return column_types


def clean_numeric_columns(df):
    """清洗数值列中的非数值字符"""
    df_clean = df.copy()

    # 尝试识别价格相关字段
    price_keywords = ['价格', '售价', '金额', '销售额', '利润', '成本']
    price_cols = [col for col in df.columns if any(kw in col for kw in price_keywords)]

    # 清洗价格相关字段
    for col in price_cols:
        if df_clean[col].dtype == 'object':
            # 去除常见非数值字符
            df_clean[col] = df_clean[col].astype(str) \
                .str.replace(r'[^\d.]', '', regex=True)
            # 转换为数值类型
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 处理百分比字段
    percent_keywords = ['率', '百分比', '占比']
    percent_cols = [col for col in df.columns if any(kw in col for kw in percent_keywords)]
    for col in percent_cols:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str) \
                .str.replace(r'[%]', '', regex=True)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce') / 100

    return df_clean


def process_categorical_variables(df, column_types, fit_encoder=True):
    """处理分类变量：有序→序数编码，无序→独热编码"""
    df_processed = df.copy()
    encoders = {}

    # 1. 有序分类：序数编码
    if column_types['ordinal'] and fit_encoder:
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_ordinal = ordinal_encoder.fit_transform(df_processed[column_types['ordinal']])
        df_ordinal = pd.DataFrame(
            df_ordinal,
            columns=[f"{col}_编码" for col in column_types['ordinal']],
            index=df_processed.index
        )
        df_processed = pd.concat([df_processed, df_ordinal], axis=1)
        encoders['ordinal'] = ordinal_encoder

    # 2. 无序分类：独热编码
    if column_types['nominal'] and fit_encoder:
        onehot_encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        df_onehot = onehot_encoder.fit_transform(df_processed[column_types['nominal']])
        # 生成独热编码字段名
        feature_names = []
        for i, col in enumerate(column_types['nominal']):
            categories = onehot_encoder.categories_[i][1:]  # 跳过第一个类别
            feature_names.extend([f"{col}_{cat}" for cat in categories])
        df_onehot = pd.DataFrame(
            df_onehot,
            columns=feature_names,
            index=df_processed.index
        )
        df_processed = pd.concat([df_processed, df_onehot], axis=1)
        encoders['onehot'] = onehot_encoder
        encoders['onehot_features'] = feature_names

    # 3. 非拟合模式
    if not fit_encoder and st.session_state.category_encoder:
        encoders = st.session_state.category_encoder
        if column_types['ordinal']:
            df_ordinal = encoders['ordinal'].transform(df_processed[column_types['ordinal']])
            df_ordinal = pd.DataFrame(
                df_ordinal,
                columns=[f"{col}_编码" for col in column_types['ordinal']],
                index=df_processed.index
            )
            df_processed = pd.concat([df_processed, df_ordinal], axis=1)
        if column_types['nominal']:
            df_onehot = encoders['onehot'].transform(df_processed[column_types['nominal']])
            df_onehot = pd.DataFrame(
                df_onehot,
                columns=encoders['onehot_features'],
                index=df_processed.index
            )
            df_processed = pd.concat([df_processed, df_onehot], axis=1)

    return df_processed, encoders

# ============================================================================
# 任务1：数据预处理类（按论文要求生成标准化输出文件）- 基于源代码重构
# ============================================================================
class Task1Preprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}
        self.column_types = None

    def step1_missing_value_analysis(self):
        """步骤1: 缺失值统计分析（生成电商 步骤1 缺失值统计结果.xlsx）"""
        # 计算缺失值统计
        rows = len(self.df)
        missing_stats = []

        for col in self.df.columns:
            non_null_count = self.df[col].count()
            missing_count = rows - non_null_count
            missing_rate = (missing_count / rows) * 100

            missing_stats.append({
                '字段名': col,
                '数据类型': str(self.df[col].dtype),
                '非空值数量': non_null_count,
                '缺失值数量': missing_count,
                '缺失比例%': round(missing_rate, 2)
            })

        missing_df = pd.DataFrame(missing_stats)
        self.results['step1_missing_stats'] = missing_df
        return missing_df

    def step2_price_processing(self, missing_stats):
        """步骤2: 进货价格处理（生成电商 步骤2 进货价格处理后数据.xlsx）"""
        import re

        df_step2 = self.df.copy()

        # 处理进货价格字段（基于源代码逻辑）
        if '进货价格' in df_step2.columns:
            # 使用正则表达式去除非数字和非小数点字符，转换为数值型
            df_step2['进货价格'] = df_step2['进货价格'].apply(
                lambda x: float(re.sub(r'[^\d\.]', '', str(x))) if re.search(r'[\d\.]', str(x)) else None
            )
            # 转换为整数型（若存在小数，四舍五入）
            df_step2['进货价格'] = df_step2['进货价格'].round().astype('Int64')  # 使用Int64支持缺失值

        self.results['step2_processed'] = df_step2
        return df_step2

    def step3_profit_correction(self, df_step2):
        """步骤3: 利润修正（生成电商 步骤3 利润修正后数据.xlsx）"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error

        df_step3 = df_step2.copy()

        # 检查必要字段是否存在
        required_cols = ['实际售价', '进货价格', '销售数', '利润']
        missing_cols = [col for col in required_cols if col not in df_step3.columns]
        if missing_cols:
            st.warning(f"利润修正需以下字段：{missing_cols}，数据中缺失，跳过利润修正")
            return df_step3

        # 计算理论利润
        df_step3['理论利润'] = (df_step3['实际售价'] - df_step3['进货价格']) * df_step3['销售数']
        # 筛选错误和正确数据
        error_data = df_step3[df_step3['利润'] != df_step3['理论利润']].copy()
        correct_data = df_step3[df_step3['利润'] == df_step3['理论利润']].copy()

        st.info(f"利润计算错误数据条数：{len(error_data)}")
        st.info(f"利润计算正确数据条数（训练数据）：{len(correct_data)}")

        if len(correct_data) == 0:
            st.warning("无利润计算正确的数据，无法训练模型进行补插，跳过利润修正")
            df_step3 = df_step3.drop(columns='理论利润')
            return df_step3

        # 准备模型训练数据
        features = ['实际售价', '进货价格', '销售数']
        X = correct_data[features]
        y = correct_data['利润']

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 1. 训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        # 评估随机森林模型
        rf_pred_test = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred_test)

        # 2. 训练KNN模型
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        # 评估KNN模型
        knn_pred_test = knn_model.predict(X_test)
        knn_mse = mean_squared_error(y_test, knn_pred_test)

        # 选择MSE较小的模型进行利润补插
        if rf_mse <= knn_mse:
            st.info(f"选择随机森林模型进行利润补插 (MSE: {rf_mse:.2f})")
            if len(error_data) > 0:
                error_X = error_data[features]
                pred_error = rf_model.predict(error_X)
                # 数据类型转换
                pred_error = pred_error.round().astype(df_step3['利润'].dtype)
                # 重置索引确保对齐
                df_step3 = df_step3.reset_index(drop=True)
                error_data = error_data.reset_index(drop=True)
                # 更新错误利润值（保持列名为"利润"）
                df_step3.loc[error_data.index, '利润'] = pred_error
        else:
            st.info(f"选择KNN模型进行利润补插 (MSE: {knn_mse:.2f})")
            if len(error_data) > 0:
                error_X = error_data[features]
                pred_error = knn_model.predict(error_X)
                # 数据类型转换
                pred_error = pred_error.round().astype(df_step3['利润'].dtype)
                # 重置索引确保对齐
                df_step3 = df_step3.reset_index(drop=True)
                error_data = error_data.reset_index(drop=True)
                # 更新错误利润值（保持列名为"利润"）
                df_step3.loc[error_data.index, '利润'] = pred_error

        # 删除临时的理论利润列
        if '理论利润' in df_step3.columns:
            df_step3 = df_step3.drop(columns='理论利润')

        self.results['step3_processed'] = df_step3
        return df_step3

    def step4_abnormal_correction(self, df_step3):
        """步骤4: 异常值修正及利润重算（生成电商 步骤4 异常修正及利润重算后数据.xlsx）"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error

        df_step4 = df_step3.copy()

        # 检查必要字段是否存在
        required_cols = ['实际售价', '进货价格', '销售数', '客户年龄']
        missing_cols = [col for col in required_cols if col not in df_step4.columns]
        if missing_cols:
            st.warning(f"异常修正需以下字段：{missing_cols}，数据中缺失，跳过异常修正")
            return df_step4

        # 标记异常数据（实际售价 < 进货价格）
        abnormal_mask = df_step4['实际售价'] < df_step4['进货价格']
        abnormal_data = df_step4[abnormal_mask].copy()
        normal_data = df_step4[~abnormal_mask].copy()

        st.info(f"成本高于售价的异常数据条数：{len(abnormal_data)}")
        st.info(f"正常数据条数（训练数据）：{len(normal_data)}")

        if len(normal_data) == 0:
            st.warning("无正常售价数据，无法训练模型进行异常修正，跳过异常修正")
            return df_step4

        # 准备模型训练数据（预测合理实际售价）
        features = ['进货价格', '销售数', '客户年龄']
        target = '实际售价'
        X = normal_data[features]
        y = normal_data[target]

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 1. 训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        # 评估随机森林模型
        rf_pred_test = rf_model.predict(X_test)
        rf_mse = mean_squared_error(y_test, rf_pred_test)

        # 2. 训练KNN模型
        knn_model = KNeighborsRegressor(n_neighbors=5)
        knn_model.fit(X_train, y_train)
        # 评估KNN模型
        knn_pred_test = knn_model.predict(X_test)
        knn_mse = mean_squared_error(y_test, knn_pred_test)

        # 综合两种模型结果进行售价补插（取平均值）
        if len(abnormal_data) > 0:
            abnormal_X = abnormal_data[features]
            rf_pred_abnormal = rf_model.predict(abnormal_X)
            knn_pred_abnormal = knn_model.predict(abnormal_X)
            combined_pred = (rf_pred_abnormal + knn_pred_abnormal) / 2
            # 数据类型转换（确保与原售价字段一致）
            combined_pred = combined_pred.round().astype(df_step4[target].dtype)
            # 更新异常数据的售价
            df_step4.loc[abnormal_mask, target] = combined_pred

        # 二次检查剩余异常（若仍有售价<进货价，将售价设为进货价）
        remaining_abnormal_mask = df_step4['实际售价'] < df_step4['进货价格']
        if remaining_abnormal_mask.sum() > 0:
            st.info(f"二次检查发现{remaining_abnormal_mask.sum()}条剩余异常数据，将售价设为进货价")
            df_step4.loc[remaining_abnormal_mask, '实际售价'] = df_step4.loc[remaining_abnormal_mask, '进货价格']

        # 重新计算正确利润（保持列名为"利润"）
        df_step4['利润'] = (df_step4['实际售价'] - df_step4['进货价格']) * df_step4['销售数']

        self.results['step4_processed'] = df_step4
        return df_step4

    def step5_standardization(self, df_step4):
        """步骤5: 标准化处理（生成电商 步骤5 MinMax标准化后数据.xlsx和ZScore标准化后数据.xlsx）"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        df_original = df_step4.copy()

        # 定义需标准化的数值列（基于源代码逻辑）
        required_cols = ["进货价格", "实际售价", "销售数", "利润"]
        # 若存在销售额列，加入标准化范围
        if "销售额" in df_original.columns:
            required_cols.append("销售额")

        # 检查列是否存在
        missing_cols = [col for col in required_cols if col not in df_original.columns]
        if missing_cols:
            st.warning(f"标准化需以下字段：{missing_cols}，数据中缺失，跳过标准化")
            return df_original, df_original

        # 筛选数值型列（排除非数值数据）
        numeric_cols = [col for col in required_cols if pd.api.types.is_numeric_dtype(df_original[col])]
        if not numeric_cols:
            st.warning("无可用的数值型列进行标准化")
            return df_original, df_original

        st.info(f"待标准化的数值列：{numeric_cols}")

        # 1. Z-Score标准化（均值为0，标准差为1）
        zscore_scaler = StandardScaler()
        df_zscore = df_original.copy()
        df_zscore[numeric_cols] = zscore_scaler.fit_transform(df_zscore[numeric_cols])

        # 2. Min-Max标准化（范围0-1）
        minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        df_minmax = df_original.copy()
        df_minmax[numeric_cols] = minmax_scaler.fit_transform(df_minmax[numeric_cols])

        self.results['step5_minmax'] = df_minmax
        self.results['step5_zscore'] = df_zscore
        self.results['numeric_cols'] = numeric_cols

        return df_minmax, df_zscore

    def generate_all_results(self):
        """生成所有步骤的结果（按论文要求的文件格式）"""
        try:
            # 执行全流程步骤
            step1_missing = self.step1_missing_value_analysis()
            step2_price = self.step2_price_processing(step1_missing)
            step3_profit = self.step3_profit_correction(step2_price)
            step4_abnormal = self.step4_abnormal_correction(step3_profit)
            step5_minmax, step5_zscore = self.step5_standardization(step4_abnormal)

            # 字段类型识别
            self.column_types = auto_detect_column_types(step4_abnormal)

            # 处理分类变量
            final_data, encoders = process_categorical_variables(
                step4_abnormal, self.column_types, fit_encoder=True)

            # 整理结果文件
            result_files = {
                '电商 步骤1 缺失值统计结果.xlsx': step1_missing,
                '电商 步骤2 进货价格处理后数据.xlsx': step2_price,
                '电商 步骤3 利润修正后数据.xlsx': step3_profit,
                '电商 步骤4 异常修正及利润重算后数据.xlsx': step4_abnormal,
                '电商 步骤5 MinMax标准化后数据.xlsx': step5_minmax,
                '电商 步骤5 ZScore标准化后数据.xlsx': step5_zscore
            }

            # 整理进度日志
            progress_log = [
                f"步骤1：完成缺失值统计，共{len(step1_missing)}个字段",
                f"步骤2：完成进货价格处理",
                f"步骤3：完成利润修正",
                f"步骤4：完成异常值修正",
                f"步骤5：完成标准化处理，生成MinMax和ZScore两种标准化结果"
            ]

            return result_files, progress_log, final_data, encoders, self.column_types

        except Exception as e:
            return None, [f"预处理错误: {str(e)}"], None, None, None

# ============================================================================
# 增强版任务2：多维销售特征分析类（按论文要求重构）- 修复热力图错误
# ============================================================================
class EnhancedTask2Analyzer:
    def __init__(self, df, column_types):
        self.df = df.copy()
        self.column_types = column_types
        self.results = {}

    def create_heatmaps(self):
        """创建热力图 - 修复数据类型问题"""
        try:
            # 使用全局中文字体配置（已在文件开头设置）

            figs = {}

            # 1. 商品品类与省份交叉热力图
            if all(col in self.df.columns for col in ['区域', '商品品类', '利润']):
                # 提取省份
                if self.df['区域'].str.contains('-').any():
                    self.df['省份'] = self.df['区域'].apply(lambda x: x.split('-')[1] if '-' in str(x) else x)
                else:
                    self.df['省份'] = self.df['区域']

                # 确保利润列是数值类型
                self.df['利润'] = pd.to_numeric(self.df['利润'], errors='coerce')

                # 过滤掉无效数据
                heatmap_data = self.df[['商品品类', '省份', '利润']].dropna()

                if len(heatmap_data) > 0:
                    plt.figure(figsize=(12, 8))
                    category_province_pivot = heatmap_data.pivot_table(
                        index='商品品类',
                        columns='省份',
                        values='利润',
                        aggfunc='sum',
                        fill_value=0
                    )

                    # 确保数据是数值类型
                    category_province_pivot = category_province_pivot.astype(float)

                    # 限制行列数量，避免热力图形状过大
                    if len(category_province_pivot) > 20:
                        category_province_pivot = category_province_pivot.head(20)
                    if len(category_province_pivot.columns) > 15:
                        category_province_pivot = category_province_pivot[category_province_pivot.columns[:15]]

                    sns.heatmap(category_province_pivot, cmap='Blues', annot=False, fmt='.0f')
                    plt.title('商品品类和省份交叉的利润热力图')
                    plt.xlabel('省份')
                    plt.xticks(rotation=45)
                    plt.ylabel('商品品类')
                    plt.tight_layout()
                    figs['category_province_profit'] = plt.gcf()
                    plt.close()
                else:
                    st.warning("商品品类-省份热力图：无有效数据")

            # 2. 省份与日期交叉热力图
            if all(col in self.df.columns for col in ['日期', '省份', '利润']):
                # 确保数据是数值类型
                self.df['利润'] = pd.to_numeric(self.df['利润'], errors='coerce')
                self.df['日期'] = pd.to_numeric(self.df['日期'], errors='coerce')

                # 过滤掉无效数据
                heatmap_data = self.df[['日期', '省份', '利润']].dropna()

                if len(heatmap_data) > 0:
                    plt.figure(figsize=(15, 8))
                    province_date_pivot = heatmap_data.pivot_table(
                        index='省份',
                        columns='日期',
                        values='利润',
                        aggfunc='sum',
                        fill_value=0
                    )

                    # 确保数据是数值类型
                    province_date_pivot = province_date_pivot.astype(float)

                    # 限制行列数量
                    if len(province_date_pivot) > 15:
                        province_date_pivot = province_date_pivot.head(15)
                    if len(province_date_pivot.columns) > 20:
                        province_date_pivot = province_date_pivot[province_date_pivot.columns[:20]]

                    sns.heatmap(province_date_pivot, cmap='Blues', annot=False, fmt='.0f')
                    plt.title('省份和日期交叉的利润热力图')
                    plt.xlabel('日期')
                    plt.xticks(rotation=90)
                    plt.ylabel('省份')
                    plt.tight_layout()
                    figs['province_date_profit'] = plt.gcf()
                    plt.close()
                else:
                    st.warning("省份-日期热力图：无有效数据")

            self.results['heatmaps'] = figs
            return len(figs) > 0

        except Exception as e:
            st.error(f"热力图生成错误: {str(e)}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
            return False

    def perform_clustering_analysis(self):
        """执行聚类分析 - 修复数据类型问题"""
        try:
            # 使用全局中文字体配置（已在文件开头设置）

            # 选择数值型列进行聚类
            numeric_cols = ['客户年龄', '进货价格', '实际售价', '销售数', '销售额', '利润']
            existing_numeric_cols = [col for col in numeric_cols if col in self.df.columns]

            if len(existing_numeric_cols) < 2:
                st.warning(f"可用于聚类的数值型列不足，仅找到: {existing_numeric_cols}")
                return False

            # 提取数值型数据并处理缺失值
            df_numeric = self.df[existing_numeric_cols].copy()

            # 确保所有列都是数值类型
            for col in existing_numeric_cols:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

            df_numeric = df_numeric.fillna(0)

            # 检查数据有效性
            if df_numeric.isnull().any().any() or (df_numeric == 0).all().any():
                st.warning("聚类数据包含无效值，跳过聚类分析")
                return False

            # 确定最佳聚类数k
            sse = []
            silhouette_scores = []
            k_range = range(2, min(11, len(df_numeric) // 2))  # 避免k值过大

            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=2024, n_init='auto')
                    kmeans.fit(df_numeric)
                    sse.append(kmeans.inertia_)
                    labels = kmeans.labels_
                    if len(set(labels)) > 1:  # 确保有多个聚类
                        score = silhouette_score(df_numeric, labels)
                        silhouette_scores.append(score)
                    else:
                        silhouette_scores.append(0)
                except Exception as e:
                    st.warning(f"聚类数k={k}时出错: {e}")
                    sse.append(0)
                    silhouette_scores.append(0)

            # 绘制评估图表
            if len(sse) > 0 and len(silhouette_scores) > 0:
                fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.plot(k_range, sse, 'bx-')
                ax1.set_xlabel('聚类数量k')
                ax1.set_ylabel('SSE（误差平方和）')
                ax1.set_title('手肘法确定最佳k值')
                ax2.plot(k_range, silhouette_scores, 'rx-')
                ax2.set_xlabel('聚类数量k')
                ax2.set_ylabel('轮廓系数')
                ax2.set_title('轮廓系数确定最佳k值')
                plt.tight_layout()

                self.results['cluster_evaluation_plot'] = fig1

                # 选择最佳k值
                if max(silhouette_scores) > 0:
                    best_k_index = silhouette_scores.index(max(silhouette_scores))
                    best_k = k_range[best_k_index]

                    # 使用最佳k值执行最终聚类
                    final_kmeans = KMeans(n_clusters=best_k, random_state=2024, n_init='auto')
                    cluster_labels = final_kmeans.fit_predict(df_numeric)

                    # 保存聚类结果
                    df_clustered = self.df.copy()
                    df_clustered['聚类标签'] = cluster_labels
                    cluster_analysis = df_clustered.groupby('聚类标签')[existing_numeric_cols].mean().round(2)

                    self.results['clustered_data'] = df_clustered
                    self.results['cluster_analysis'] = cluster_analysis
                    self.results['best_k'] = best_k

                    return True
                else:
                    st.warning("无法确定有效的最佳k值，跳过聚类")
                    return False
            else:
                st.warning("聚类评估数据不足，跳过聚类分析")
                return False

        except Exception as e:
            st.error(f"聚类分析错误: {str(e)}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
            return False

    def generate_city_distribution_data(self):
        """生成城市分布数据（对应论文图4）"""
        if '区域' not in self.df.columns:
            return None

        # 提取城市信息
        if self.df['区域'].str.contains('-').any():
            self.df['城市'] = self.df['区域'].apply(lambda x: x.split('-')[1] if '-' in str(x) else x)
        else:
            self.df['城市'] = self.df['区域']

        # 城市用户数统计
        city_stats = self.df['城市'].value_counts().reset_index()
        city_stats.columns = ['城市', '用户数']
        city_stats = city_stats.head(15)  # Top 15城市

        return city_stats

    def generate_province_distribution_data(self):
        """生成省份分布数据（对应论文图5）"""
        if '区域' not in self.df.columns:
            return None

        # 提取省份信息
        if '省份' not in self.df.columns:
            if self.df['区域'].str.contains('-').any():
                self.df['省份'] = self.df['区域'].apply(lambda x: x.split('-')[0] if '-' in str(x) else x)
            else:
                self.df['省份'] = self.df['区域']

        province_stats = self.df['省份'].value_counts().reset_index()
        province_stats.columns = ['省份', '用户数']

        return province_stats

    def generate_city_tier_data(self):
        """生成城市分级数据（对应论文图6）"""
        if '城市' not in self.df.columns:
            return None

        # 城市分级定义（根据论文）
        tier_1 = ['北京', '上海', '广州', '深圳']
        tier_2 = ['昆明', '福州', '厦门', '无锡', '哈尔滨', '长春', '宁波', '济南', '大连', '郑州',
                  '长沙', '成都', '杭州', '南京', '武汉', '西安', '苏州', '天津', '青岛', '沈阳',
                  '东莞', '佛山', '合肥', '石家庄', '南宁', '常州', '烟台', '唐山', '徐州', '温州']
        tier_3 = ['兰州', '海口', '乌鲁木齐', '贵阳', '银川', '西宁', '呼和浩特', '拉萨', '保定',
                  '惠州', '珠海', '中山', '江门', '肇庆', '清远', '韶关', '湛江', '茂名', '阳江',
                  '云浮', '汕头', '潮州', '揭阳', '汕尾', '梅州', '河源']

        def assign_city_tier(city):
            if pd.isna(city):
                return '其他城市'
            city_str = str(city)
            if city_str in tier_1:
                return '一线城市'
            elif city_str in tier_2:
                return '二线城市'
            elif city_str in tier_3:
                return '三线城市'
            else:
                return '其他城市'

        self.df['城市等级'] = self.df['城市'].apply(assign_city_tier)
        tier_stats = self.df['城市等级'].value_counts().reset_index()
        tier_stats.columns = ['城市等级', '用户数']
        tier_stats['占比'] = (tier_stats['用户数'] / len(self.df) * 100).round(2)

        return tier_stats

    def generate_region_tier_data(self):
        """生成区域分级数据（对应论文图7）"""
        if '省份' not in self.df.columns:
            return None

        # 区域定义
        region_mapping = {
            '华南': ['广东', '广西', '海南', '福建'],
            '华东': ['上海', '江苏', '浙江', '安徽', '江西', '山东'],
            '华北': ['北京', '天津', '河北', '山西', '内蒙古'],
            '东北': ['辽宁', '吉林', '黑龙江'],
            '西南': ['重庆', '四川', '贵州', '云南', '西藏'],
            '西北': ['陕西', '甘肃', '青海', '宁夏', '新疆'],
            '华中': ['河南', '湖北', '湖南']
        }

        def assign_region(province):
            if pd.isna(province):
                return '其他'
            province_str = str(province)
            for region, provinces in region_mapping.items():
                if province_str in provinces:
                    return region
            return '其他'

        self.df['大区'] = self.df['省份'].apply(assign_region)
        region_stats = self.df['大区'].value_counts().reset_index()
        region_stats.columns = ['大区', '用户数']
        region_stats['占比'] = (region_stats['用户数'] / len(self.df) * 100).round(2)

        return region_stats

    def generate_gender_category_analysis(self):
        """生成性别-品类分析数据（对应论文图8）"""
        if not all(col in self.df.columns for col in ['客户性别', '商品品类']):
            return None

        gender_category_stats = self.df.groupby(['商品品类', '客户性别']).size().reset_index()
        gender_category_stats.columns = ['商品品类', '客户性别', '订单人数']

        return gender_category_stats

    def generate_age_gender_analysis(self):
        """生成年龄-性别分析数据（对应论文图9）"""
        if '客户年龄' not in self.df.columns or '客户性别' not in self.df.columns:
            return None

        # 确保年龄是数值类型
        self.df['客户年龄'] = pd.to_numeric(self.df['客户年龄'], errors='coerce')

        # 年龄分段
        def assign_age_group(age):
            if pd.isna(age):
                return '未知'
            try:
                age = int(age)
                if age < 25:
                    return '20-24岁'
                elif age < 30:
                    return '25-29岁'
                elif age < 35:
                    return '30-34岁'
                elif age < 40:
                    return '35-39岁'
                elif age < 45:
                    return '40-44岁'
                elif age < 50:
                    return '45-49岁'
                elif age < 55:
                    return '50-54岁'
                elif age < 60:
                    return '55-59岁'
                else:
                    return '60岁以上'
            except:
                return '未知'

        self.df['年龄段'] = self.df['客户年龄'].apply(assign_age_group)
        age_gender_stats = self.df.groupby(['年龄段', '客户性别']).size().reset_index()
        age_gender_stats.columns = ['年龄段', '客户性别', '订单人数']

        return age_gender_stats

    def generate_time_series_analysis(self):
        """生成时间序列分析数据（对应论文图10）"""
        date_col = next((col for col in self.column_types['identifier'] if '日期' in col), None)
        if not date_col:
            return None

        # 确保日期是数值类型
        self.df[date_col] = pd.to_numeric(self.df[date_col], errors='coerce')

        time_stats = self.df.groupby(date_col).size().reset_index()
        time_stats.columns = ['日期', '订单人数总和']

        return time_stats

    def generate_correlation_analysis(self):
        """生成相关性分析数据（对应图13）"""
        numeric_cols = self.column_types['numeric']
        if len(numeric_cols) < 2:
            return None

        # 确保所有数值列都是数值类型
        correlation_data = self.df[numeric_cols].copy()
        for col in numeric_cols:
            correlation_data[col] = pd.to_numeric(correlation_data[col], errors='coerce')

        correlation_data = correlation_data.dropna()

        if len(correlation_data) < 2:
            return None

        correlation_matrix = correlation_data.corr().round(4)

        return correlation_matrix

    def generate_all_analysis_data(self):
        """生成所有分析维度的数据"""
        analysis_results = {}

        # 地理分布分析
        analysis_results['city_distribution'] = self.generate_city_distribution_data()
        analysis_results['province_distribution'] = self.generate_province_distribution_data()
        analysis_results['city_tier_analysis'] = self.generate_city_tier_data()
        analysis_results['region_tier_analysis'] = self.generate_region_tier_data()

        # 客户画像分析
        analysis_results['gender_category_analysis'] = self.generate_gender_category_analysis()
        analysis_results['age_gender_analysis'] = self.generate_age_gender_analysis()

        # 时间序列分析
        analysis_results['time_series_analysis'] = self.generate_time_series_analysis()

        # 统计关系分析
        analysis_results['correlation_analysis'] = self.generate_correlation_analysis()

        # 保留原有的热力图和聚类分析
        analysis_results.update(self.results)

        return analysis_results

def show_python_visualizations(analyzer):
    """显示Python原生可视化"""
    st.subheader("Python可视化展示")

    # 原有的热力图和聚类分析展示
    if 'heatmaps' in analyzer.results and len(analyzer.results['heatmaps']) > 0:
        st.subheader("1. 交叉维度热力图分析")
        for name, fig in analyzer.results['heatmaps'].items():
            st.pyplot(fig)

    if 'cluster_evaluation_plot' in analyzer.results:
        st.subheader("2. 聚类分析结果")
        st.pyplot(analyzer.results['cluster_evaluation_plot'])

        if 'cluster_analysis' in analyzer.results:
            st.subheader("聚类特征平均值对比")
            st.dataframe(analyzer.results['cluster_analysis'])

    # 确保analyzer对象有效
    if not analyzer or not hasattr(analyzer, 'results'):
        st.error("分析器对象无效，无法显示可视化结果")
        return

    # 调用字典版本函数
    show_python_visualizations_from_dict(analyzer.results)


def show_python_visualizations_from_dict(results_dict):
    """从字典数据显示Python可视化 - 用于处理序列化后的数据"""
    st.subheader("Python可视化展示")

    # 确保结果字典有效
    if not results_dict:
        st.error("分析结果数据无效，无法显示可视化结果")
        return

    # 热力图显示
    if 'heatmaps' in results_dict and len(results_dict['heatmaps']) > 0:
        st.subheader("1. 交叉维度热力图分析")
        for name, fig in results_dict['heatmaps'].items():
            try:
                st.pyplot(fig)
                plt.close(fig)  # 关闭图形释放内存
            except Exception as e:
                st.error(f"显示热力图 {name} 时出错: {str(e)}")
    else:
        st.info("暂无热力图数据")

    # 聚类分析显示
    if 'cluster_evaluation_plot' in results_dict:
        st.subheader("2. 聚类分析结果")
        try:
            st.pyplot(results_dict['cluster_evaluation_plot'])
            plt.close(results_dict['cluster_evaluation_plot'])
        except Exception as e:
            st.error(f"显示聚类评估图时出错: {str(e)}")

        if 'cluster_analysis' in results_dict:
            st.subheader("聚类特征平均值对比")
            st.dataframe(results_dict['cluster_analysis'])
    else:
        st.info("暂无聚类分析数据")

    # 其他分析结果显示
    if 'sensitivity_plot' in results_dict:
        st.subheader("3. 价格敏感度分析")
        try:
            st.pyplot(results_dict['sensitivity_plot'])
            plt.close(results_dict['sensitivity_plot'])
        except Exception as e:
            st.error(f"显示价格敏感度图时出错: {str(e)}")


def show_python_visualizations(analyzer):
    """显示Python原生可视化 - 用于第一次执行时的对象"""
    st.subheader("Python可视化展示")

    # 确保analyzer对象有效
    if not analyzer or not hasattr(analyzer, 'results'):
        st.error("分析器对象无效，无法显示可视化结果")
        return

    # 调用字典版本函数
    show_python_visualizations_from_dict(analyzer.results)


def show_data_export_interface(analysis_data):
    """显示数据导出界面 - 使用Excel格式避免编码问题"""
    st.subheader("图表数据导出")

    st.markdown("""
    ### 导出说明
    以下数据可直接用于在Excel、Tableau、Echarts等工具中制作图表。
    为避免编码问题，已提供Excel格式下载。
    """)

    def convert_to_excel(df, sheet_name="数据"):
        """将DataFrame转换为Excel格式"""
        import io
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        output.seek(0)
        return output.getvalue()

    # 地理分布数据导出
    st.markdown("#### 地理分布分析")

    if analysis_data.get('city_distribution') is not None:
        excel_data = convert_to_excel(analysis_data['city_distribution'], "城市分布")
        st.download_button(
            label="下载城市分布数据",
            data=excel_data,
            file_name="城市分布数据_Top15城市.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if analysis_data.get('province_distribution') is not None:
        excel_data = convert_to_excel(analysis_data['province_distribution'], "省份分布")
        st.download_button(
            label="下载省份分布数据",
            data=excel_data,
            file_name="省份分布数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if analysis_data.get('city_tier_analysis') is not None:
        excel_data = convert_to_excel(analysis_data['city_tier_analysis'], "城市分级")
        st.download_button(
            label="下载城市分级数据",
            data=excel_data,
            file_name="城市分级环状图数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if analysis_data.get('region_tier_analysis') is not None:
        excel_data = convert_to_excel(analysis_data['region_tier_analysis'], "区域分级")
        st.download_button(
            label="下载区域分级数据",
            data=excel_data,
            file_name="区域分级环状图数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 客户画像数据导出
    st.markdown("#### 客户画像分析")

    if analysis_data.get('gender_category_analysis') is not None:
        excel_data = convert_to_excel(analysis_data['gender_category_analysis'], "性别品类")
        st.download_button(
            label="下载性别-品类数据",
            data=excel_data,
            file_name="性别品类交叉分析数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if analysis_data.get('age_gender_analysis') is not None:
        excel_data = convert_to_excel(analysis_data['age_gender_analysis'], "年龄性别")
        st.download_button(
            label="下载年龄-性别数据",
            data=excel_data,
            file_name="年龄性别分布数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 时间序列和相关性分析
    st.markdown("#### 时间与关系分析")

    if analysis_data.get('time_series_analysis') is not None:
        excel_data = convert_to_excel(analysis_data['time_series_analysis'], "时间序列")
        st.download_button(
            label="下载时间序列数据",
            data=excel_data,
            file_name="时间序列订单数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if analysis_data.get('correlation_analysis') is not None:
        # 相关性矩阵需要保留索引
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            analysis_data['correlation_analysis'].to_excel(writer, sheet_name="相关性矩阵")
        excel_data = output.getvalue()
        st.download_button(
            label="下载相关性矩阵",
            data=excel_data,
            file_name="变量相关性矩阵.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # 数据预览（保持不变）
    st.markdown("#### 👀 数据预览")
    available_datasets = [key for key in analysis_data.keys() if
                          analysis_data[key] is not None and hasattr(analysis_data[key], 'head')]
    if available_datasets:
        dataset_to_preview = st.selectbox(
            "选择要预览的数据集:",
            available_datasets
        )

        if dataset_to_preview:
            st.dataframe(analysis_data[dataset_to_preview].head(10))

            # 数据统计信息
            st.markdown("**数据统计:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("行数", len(analysis_data[dataset_to_preview]))
            with col2:
                st.metric("列数", len(analysis_data[dataset_to_preview].columns))
            with col3:
                st.metric("数据类型", str(analysis_data[dataset_to_preview].dtypes.unique()[0]))




# ============================================================================
# 任务3：销售预测类（修复版 - 与源代码保持一致）
# ============================================================================
class Task3Forecaster:
    def __init__(self, df, column_types):
        self.df = df.copy()
        self.column_types = column_types
        self.results = {}

    def prepare_time_series_data(self):
        """使用源代码的数据准备逻辑 - 修复数据类型问题"""
        try:
            # 使用源代码的直接转换方式
            date_col = next((col for col in self.column_types['identifier'] if '日期' in col), None)
            if not date_col:
                st.error("未识别到日期字段，无法构建时间序列")
                return False

            # 改为源代码的转换方式
            self.df[date_col] = self.df[date_col].astype(int)

            # 确保利润列是数值类型（源代码方式）
            self.df['利润'] = pd.to_numeric(self.df['利润'], errors='coerce')
            self.df = self.df.dropna(subset=['利润'])

            # 按日聚合利润数据
            daily_profit = self.df.groupby(date_col)['利润'].sum().reset_index()
            daily_profit = daily_profit.rename(columns={'利润': '每日总利润'})

            # 划分训练集和测试集
            train = daily_profit[daily_profit[date_col] <= 24]
            test = daily_profit[daily_profit[date_col] > 24]

            if len(train) == 0 or len(test) == 0:
                st.error("数据日期范围不足，无法划分训练测试集")
                return False

            # 🔥 关键修复：确保数据是标准的numpy数组，不是IntegerArray
            self.results['time_series_data'] = daily_profit
            self.results['train_data'] = train
            self.results['test_data'] = test
            self.results['date_col'] = date_col

            # 转换为标准的numpy数组，避免IntegerArray问题
            self.results['y_train'] = train['每日总利润'].values.astype(float)
            self.results['y_test'] = test['每日总利润'].values.astype(float)

            st.success(f"时间序列准备完成：训练集{len(train)}天，测试集{len(test)}天")
            return True

        except Exception as e:
            st.error(f"时间序列准备错误: {str(e)}")
            return False

    def create_features(self, day_indices, residuals=None):
        """使用源代码的特征工程逻辑"""
        features = []

        # 预计算每个日期的统计量（源代码逻辑）
        daily_stats = self.df.groupby(self.results['date_col']).agg({
            '销售额': ['count', 'mean', 'sum'],
            '实际售价': 'mean',
            '进货价格': 'mean',
            '客户性别': lambda x: (x == '女').mean()
        }).round(4)

        daily_stats.columns = ['order_count', 'avg_sale', 'total_sale',
                               'avg_selling_price', 'avg_cost_price', 'female_ratio']

        # 源代码的毛利率计算（不处理除0）
        daily_stats['gross_profit_margin'] = (
                (daily_stats['avg_selling_price'] - daily_stats['avg_cost_price']) /
                daily_stats['avg_cost_price']
        ).fillna(0).round(4)

        # 源代码的单客价值计算
        daily_stats['customer_value'] = (
                daily_stats['total_sale'] / daily_stats['order_count']
        ).fillna(0).round(2)

        # 训练集统计量（用于填充缺失值）- 源代码逻辑
        train_days_data = self.df[self.df[self.results['date_col']] <= 24]
        train_stats = train_days_data.groupby(self.results['date_col']).agg({
            '销售额': ['count', 'mean', 'sum'],
            '客户性别': lambda x: (x == '女').mean()
        })
        train_stats.columns = ['order_count', 'avg_sale', 'total_sale', 'female_ratio']

        for day in day_indices:
            day_features = {}

            # 1. 基础时间特征（与源代码一致）
            day_features['day'] = int(day)
            day_features['day_of_week'] = (int(day) - 1) % 7
            day_features['day_of_month'] = int(day)
            day_features['is_weekend'] = 1 if day_features['day_of_week'] in [5, 6] else 0
            day_features['is_month_end'] = 1 if int(day) >= 28 else 0

            # 2. 从预计算的统计量中获取业务特征（源代码逻辑）
            if day in daily_stats.index:
                stats = daily_stats.loc[day]
                day_features.update({
                    'order_count': float(stats['order_count']),
                    'avg_sale_amount': float(stats['avg_sale']),
                    'total_sale': float(stats['total_sale']),
                    'gross_profit_margin': float(stats['gross_profit_margin']),
                    'customer_value': float(stats['customer_value']),
                    'female_ratio': float(stats['female_ratio'])
                })
            else:
                # 使用源代码的中位数填充逻辑
                day_features.update({
                    'order_count': float(train_stats['order_count'].median()),
                    'avg_sale_amount': float(train_stats['avg_sale'].median()),
                    'total_sale': float(train_stats['total_sale'].median()),
                    'gross_profit_margin': float(0.3),  # 源代码的默认值
                    'customer_value': float(
                        train_stats['total_sale'].median() / max(1, train_stats['order_count'].median())),
                    'female_ratio': float(train_stats['female_ratio'].median())
                })

            # 3. 滞后残差特征（源代码逻辑）
            if residuals is not None:
                for lag in [1, 2, 3]:
                    lag_day = int(day) - lag
                    lag_key = f'residual_lag_{lag}'
                    if lag_day > 0 and lag_day in residuals.index:
                        day_features[lag_key] = float(residuals[lag_day])
                    else:
                        day_features[lag_key] = float(residuals.median() if not residuals.empty else 0)

            features.append(day_features)

        return pd.DataFrame(features)

    def hybrid_forecast(self):
        """使用源代码的ARIMA-XGBoost混合预测逻辑 - 修复数据类型"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from xgboost import XGBRegressor
            from sklearn.metrics import mean_absolute_percentage_error

            # 获取数据
            train = self.results['train_data']
            test = self.results['test_data']
            y_train = self.results['y_train']  # 已经是float数组
            y_test = self.results['y_test']  # 已经是float数组
            date_col = self.results['date_col']

            # 1. ARIMA建模 - 使用源代码参数 (2,1,2)
            st.info("Step 1: ARIMA建模...")
            try:
                # 🔥 确保y_train是标准的numpy float数组
                y_train_arima = y_train.astype(float)

                arima_model = ARIMA(y_train_arima, order=(2, 1, 2))  # 改为源代码参数
                arima_fit = arima_model.fit()
                arima_train_pred = arima_fit.predict(start=0, end=len(y_train_arima) - 1)
                arima_test_pred = arima_fit.forecast(steps=len(y_test))
                st.success(f"ARIMA模型训练成功 (AIC: {arima_fit.aic:.2f})")
            except Exception as e:
                st.warning(f"ARIMA模型训练失败，使用均值预测: {e}")
                # 使用numpy数组避免数据类型问题
                arima_train_pred = np.full_like(y_train, float(np.mean(y_train)))
                arima_test_pred = np.full_like(y_test, float(np.mean(y_train)))
                arima_fit = None

            # 2. 计算残差 - 确保是float类型
            residuals_train = y_train.astype(float) - arima_train_pred.astype(float)
            residual_series = pd.Series(residuals_train, index=train[date_col].values)

            # 3. XGBoost学习残差 - 使用源代码参数
            st.info("Step 2: XGBoost学习残差...")

            # 创建特征
            X_train = self.create_features(train[date_col].values, residual_series)
            X_train = X_train.fillna(0)

            # 确保特征都是数值类型
            for col in X_train.columns:
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
            X_train = X_train.fillna(0)

            # XGBoost模型训练 - 源代码参数
            xgb_model = XGBRegressor(
                max_depth=3,
                learning_rate=0.05,
                n_estimators=1000,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                eval_metric='mae'
            )
            xgb_model.fit(X_train, residuals_train)

            # 测试集特征
            X_test = self.create_features(test[date_col].values)
            X_test = X_test.fillna(0)

            # 确保特征列一致
            for col in X_train.columns:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[X_train.columns]

            # 确保测试集特征都是数值类型
            for col in X_test.columns:
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
            X_test = X_test.fillna(0)

            # 预测残差
            xgb_residual_pred = xgb_model.predict(X_test)

            # 4. 最终预测
            final_pred = arima_test_pred.astype(float) + xgb_residual_pred.astype(float)
            mape = mean_absolute_percentage_error(y_test, final_pred) * 100

            # 保存结果
            self.results['arima_model'] = arima_fit
            self.results['xgb_model'] = xgb_model
            self.results['arima_test_pred'] = arima_test_pred.astype(float)
            self.results['xgb_residual_pred'] = xgb_residual_pred.astype(float)
            self.results['final_pred'] = final_pred.astype(float)
            self.results['mape'] = mape
            self.results['residuals_train'] = residuals_train.astype(float)
            self.results['feature_importance'] = pd.DataFrame({
                'feature': X_train.columns,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False)

            # 创建详细结果表
            results_df = pd.DataFrame({
                '日期': test[date_col].values,
                '实际利润': y_test,
                'ARIMA预测': arima_test_pred.astype(float),
                'XGBoost残差预测': xgb_residual_pred.astype(float),
                '最终预测': final_pred.astype(float),
                '相对误差(%)': (np.abs(y_test - final_pred) / y_test * 100).astype(float)
            })
            self.results['detailed_results'] = results_df

            st.success(f"混合预测完成！测试集MAPE: {mape:.2f}%")
            return True

        except Exception as e:
            st.error(f"混合预测错误: {str(e)}")
            import traceback
            st.error(f"详细错误: {traceback.format_exc()}")
            return False

    def generate_visualizations(self):
        """生成可视化图表 - 保持不变"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置中文字体
            # 使用全局中文字体配置（已在文件开头设置）

            figs = {}

            # 1. 主预测对比图
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            train = self.results['train_data']
            test = self.results['test_data']
            date_col = self.results['date_col']
            y_train = self.results['y_train']
            y_test = self.results['y_test']

            # 绘制训练集实际值
            ax1.plot(train[date_col], y_train / 10000, 'bo-', label='训练集实际值',
                     alpha=0.7, markersize=6, linewidth=2)
            # 绘制测试集实际值
            ax1.plot(test[date_col], y_test / 10000, 'ro-', label='测试集实际值',
                     alpha=0.7, markersize=8, linewidth=2)

            # 绘制ARIMA训练集拟合值（源代码中的图表）
            try:
                arima_train_fit = self.results['arima_model'].predict(start=1, end=24)
                ax1.plot(train[date_col], arima_train_fit / 10000, 'c--',
                         label='ARIMA训练集拟合', alpha=0.8, linewidth=2)
            except:
                pass

            # 绘制ARIMA测试集预测值
            ax1.plot(test[date_col], self.results['arima_test_pred'] / 10000, 'm--',
                     label='ARIMA测试集预测', alpha=0.8, linewidth=2)
            # 绘制最终组合预测值
            ax1.plot(test[date_col], self.results['final_pred'] / 10000, 'gs-',
                     label='ARIMA+XGBoost最终预测', markersize=8, linewidth=2)

            ax1.set_xlabel('日期 (11月天数)', fontsize=12)
            ax1.set_ylabel('利润 (万元)', fontsize=12)
            ax1.set_title('利润预测对比图', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=24.5, color='gray', linestyle=':', alpha=0.7, linewidth=2)
            ax1.text(24.7, ax1.get_ylim()[1] * 0.9, '测试集开始', rotation=90, va='top', fontsize=10)
            figs['main_forecast'] = fig1

            # 2. 误差分析图
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            relative_errors = self.results['detailed_results']['相对误差(%)']
            bars = ax2.bar(test[date_col], relative_errors, alpha=0.7, color='orange',
                           edgecolor='darkorange', linewidth=1)

            ax2.set_xlabel('日期 (11月天数)', fontsize=12)
            ax2.set_ylabel('相对误差 (%)', fontsize=12)
            ax2.set_title(f'预测误差分析 (MAPE = {self.results["mape"]:.2f}%)',
                          fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # 添加误差数值标签
            for date, error in zip(test[date_col], relative_errors):
                ax2.text(date, error + 1, f'{error:.1f}%', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')
            figs['error_analysis'] = fig2

            # 3. 残差分析图
            fig3, ax3 = plt.subplots(figsize=(12, 6))

            if 'residuals_train' in self.results:
                residuals_train = self.results['residuals_train']

                # 绘制残差
                train_dates = train[date_col].values
                ax3.plot(train_dates, residuals_train, 'o-', color='purple',
                         alpha=0.7, markersize=6, linewidth=2, label='每日残差')

                # 零基准线
                ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='零基准线')

                # 均值线
                mean_residual = residuals_train.mean()
                ax3.axhline(y=mean_residual, color='blue', linestyle=':', linewidth=2, alpha=0.7,
                            label=f'均值: {mean_residual:.2f}')

                ax3.set_xlabel('训练集日期 (11月天数)', fontsize=12)
                ax3.set_ylabel('残差值', fontsize=12)
                ax3.set_title('ARIMA模型残差分布', fontsize=14, fontweight='bold')
                ax3.legend(fontsize=11)
                ax3.grid(True, alpha=0.3)
                ax3.set_xticks(train_dates)

                # 统计信息框
                stats_text = (f'均值: {residuals_train.mean():.2f}\n'
                              f'标准差: {residuals_train.std():.2f}\n'
                              f'最大值: {residuals_train.max():.2f}\n'
                              f'最小值: {residuals_train.min():.2f}')

                ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, fontsize=11,
                         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                            facecolor="lightgray", alpha=0.7))

                figs['residual_analysis'] = fig3

            # 4. 特征重要性图
            fig4, ax4 = plt.subplots(figsize=(12, 8))

            if 'feature_importance' in self.results:
                feature_importance = self.results['feature_importance'].head(10)

                # 特征名称映射
                feature_names_map = {
                    'day': '日期', 'day_of_week': '星期', 'day_of_month': '月内天数',
                    'is_weekend': '是否周末', 'is_month_end': '是否月末',
                    'order_count': '订单数', 'avg_sale_amount': '平均销售额',
                    'total_sale': '总销售额', 'gross_profit_margin': '毛利率',
                    'customer_value': '单客价值', 'female_ratio': '女性比例',
                    'residual_lag_1': '残差滞后1天', 'residual_lag_2': '残差滞后2天',
                    'residual_lag_3': '残差滞后3天'
                }

                feature_importance['feature_cn'] = feature_importance['feature'].map(
                    lambda x: feature_names_map.get(x, x)
                )
                feature_importance = feature_importance.sort_values('importance', ascending=True)

                y_pos = np.arange(len(feature_importance))
                colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))

                ax4.barh(y_pos, feature_importance['importance'], color=colors,
                         alpha=0.8, edgecolor='black')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(feature_importance['feature_cn'], fontsize=11)
                ax4.set_xlabel('特征重要性得分', fontsize=12, fontweight='bold')
                ax4.set_title('XGBoost特征重要性排名', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='x')

                # 添加重要性数值标签
                for i, v in enumerate(feature_importance['importance']):
                    ax4.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10, fontweight='bold')

                figs['feature_importance'] = fig4

            self.results['visualizations'] = figs
            return True

        except Exception as e:
            st.error(f"可视化生成错误: {str(e)}")
            return False

    def generate_all_results(self, forecast_days=14):
        """生成所有预测结果"""
        try:
            if not self.prepare_time_series_data():
                return None, ["时间序列数据准备失败"]

            if not self.hybrid_forecast():
                return None, ["混合预测模型执行失败"]

            if not self.generate_visualizations():
                return None, ["可视化生成失败"]

            # 整理结果文件
            result_files = {
                '01_时间序列历史数据.xlsx': self.results['time_series_data'],
                '02_销售预测结果.xlsx': self.results['detailed_results'],
                '03_特征重要性分析.xlsx': self.results['feature_importance']
            }

            # 进度日志
            progress_log = [
                f"时间序列准备完成：训练集{len(self.results['train_data'])}天，测试集{len(self.results['test_data'])}天",
                f"ARIMA-XGBoost混合预测完成：测试集MAPE {self.results['mape']:.2f}%",
                f"特征重要性分析完成：{len(self.results['feature_importance'])}个特征",
                f"可视化图表生成完成：{len(self.results['visualizations'])}个分析图表"
            ]

            return result_files, progress_log

        except Exception as e:
            return None, [f"预测错误: {str(e)}"]
# ============================================================================
# 任务4：运营策略优化类
# ============================================================================
class Task4Optimizer:
    def __init__(self, df, column_types):
        self.df = df.copy()
        self.column_types = column_types
        self.results = {}

    def abc_analysis(self):
        """ABC分类分析"""
        try:
            # 1. 按商品品类ABC分类
            if all(x in self.df.columns for x in ['商品品类', '销售额', '利润']):
                category_metrics = self.df.groupby('商品品类').agg({
                    '销售额': 'sum',
                    '利润': 'sum',
                    '销售数': 'count'
                }).reset_index()
                category_metrics = category_metrics.sort_values('销售额', ascending=False)
                category_metrics['销售额累计占比%'] = (
                        category_metrics['销售额'].cumsum() / category_metrics['销售额'].sum() * 100).round(2)
                category_metrics['利润累计占比%'] = (
                        category_metrics['利润'].cumsum() / category_metrics['利润'].sum() * 100).round(2)

                # ABC分类规则
                def assign_abc(cumulative_percent):
                    if cumulative_percent <= 70:
                        return 'A类（核心）'
                    elif cumulative_percent <= 90:
                        return 'B类（潜力）'
                    else:
                        return 'C类（长尾）'

                category_metrics['ABC分类（按销售额）'] = category_metrics['销售额累计占比%'].apply(assign_abc)
                category_metrics['ABC分类（按利润）'] = category_metrics['利润累计占比%'].apply(assign_abc)

                # 2. 按区域ABC分类
                if '区域' in self.df.columns:
                    region_metrics = self.df.groupby('区域').agg({
                        '销售额': 'sum',
                        '利润': 'sum'
                    }).reset_index()
                    region_metrics = region_metrics.sort_values('销售额', ascending=False)
                    region_metrics['销售额累计占比%'] = (
                            region_metrics['销售额'].cumsum() / region_metrics['销售额'].sum() * 100).round(2)
                    region_metrics['ABC分类（按销售额）'] = region_metrics['销售额累计占比%'].apply(assign_abc)

                    self.results['region_abc'] = region_metrics

                self.results['category_abc'] = category_metrics
                return True
            st.warning("缺少ABC分类所需字段（商品品类、销售额、利润）")
            return False
        except Exception as e:
            st.error(f"ABC分类错误: {str(e)}")
            return False

    def price_sensitivity_analysis(self):
        """价格敏感度分析"""
        try:
            if not all(x in self.df.columns for x in ['商品品类', '实际售价', '销售数']):
                st.warning("缺少价格敏感度分析所需字段（商品品类、实际售价、销售数）")
                return False

            sensitivity_results = []
            # 1. 按品类分析价格敏感度
            for category in self.df['商品品类'].unique():
                category_data = self.df[self.df['商品品类'] == category].copy()
                if len(category_data) < 10:
                    st.info(f"商品品类【{category}】样本量不足10条，跳过分析")
                    continue

                # 等频8区间划分
                category_data['价格区间'] = pd.qcut(
                    category_data['实际售价'],
                    q=8,
                    labels=[f'区间{i}' for i in range(1, 9)],
                    duplicates='drop'
                )
                price_sales = category_data.groupby('价格区间').agg({
                    '实际售价': 'mean',
                    '销售数': 'sum'
                }).reset_index()

                # 计算敏感度系数
                slope, intercept, r_value, p_value, std_err = linregress(
                    price_sales['实际售价'],
                    price_sales['销售数']
                )
                sensitivity_coeff = slope / (price_sales['销售数'].mean() / price_sales['实际售价'].mean())

                # 敏感度等级判定
                if sensitivity_coeff < -0.3:
                    level = '高敏感度（价格主导）'
                elif sensitivity_coeff < -0.1:
                    level = '中敏感度（价格+品质）'
                else:
                    level = '低敏感度（品质主导）'

                sensitivity_results.append({
                    '分析维度': '商品品类',
                    '维度值': category,
                    '价格弹性系数': sensitivity_coeff.round(4),
                    'R²（拟合优度）': round(r_value ** 2, 4),
                    '敏感度等级': level,
                    '样本量': len(category_data)
                })

            # 2. 按人群分析价格敏感度
            if '客户性别' in self.df.columns:
                st.info("开始按客户性别分析价格敏感度")
                for gender in self.df['客户性别'].unique():
                    gender_data = self.df[self.df['客户性别'] == gender].copy()
                    if len(gender_data) < 20:
                        st.info(f"客户性别【{gender}】样本量不足20条，跳过分析")
                        continue

                    gender_data['价格区间'] = pd.qcut(
                        gender_data['实际售价'],
                        q=8,
                        labels=[f'区间{i}' for i in range(1, 9)],
                        duplicates='drop'
                    )
                    price_sales = gender_data.groupby('价格区间').agg({
                        '实际售价': 'mean',
                        '销售数': 'sum'
                    }).reset_index()

                    slope, intercept, r_value, p_value, std_err = linregress(
                        price_sales['实际售价'],
                        price_sales['销售数']
                    )
                    sensitivity_coeff = slope / (price_sales['销售数'].mean() / price_sales['实际售价'].mean())

                    level = '高敏感度' if sensitivity_coeff < -0.3 else '中敏感度' if sensitivity_coeff < -0.1 else '低敏感度'
                    sensitivity_results.append({
                        '分析维度': '客户性别',
                        '维度值': gender,
                        '价格弹性系数': sensitivity_coeff.round(4),
                        'R²（拟合优度）': round(r_value ** 2, 4),
                        '敏感度等级': level,
                        '样本量': len(gender_data)
                    })

            sensitivity_df = pd.DataFrame(sensitivity_results)
            self.results['price_sensitivity'] = sensitivity_df
            st.success("价格敏感度分析完成")

            # 可视化：高敏感度品类TOP5
            high_sensitivity = sensitivity_df[sensitivity_df['分析维度'] == '商品品类'].nsmallest(5, '价格弹性系数')
            if len(high_sensitivity) > 0:
                fig, ax = plt.subplots(figsize=(15, 8))
                sns.barplot(x='维度值', y='价格弹性系数', data=high_sensitivity, palette='Reds')
                ax.set_title('商品品类价格敏感度TOP5（弹性系数越小越敏感）', fontsize=14)
                ax.set_xlabel('商品品类', fontsize=12)
                ax.set_ylabel('价格弹性系数', fontsize=12)
                ax.axhline(y=-0.3, color='red', linestyle='--', alpha=0.7, label='高敏感度阈值（-0.3）')
                ax.legend()
                plt.xticks(rotation=45)
                self.results['sensitivity_plot'] = fig
                st.pyplot(fig)

            return True
        except Exception as e:
            st.error(f"价格敏感度分析错误: {str(e)}")
            return False

    def generate_operation_strategy(self):
        """生成运营策略"""
        try:
            strategies = []
            # 1. 商品品类策略
            if 'category_abc' in self.results and 'price_sensitivity' in self.results:
                category_abc = self.results['category_abc']
                price_sensitivity = self.results['price_sensitivity'][
                    self.results['price_sensitivity']['分析维度'] == '商品品类']

                for _, abc_row in category_abc.iterrows():
                    category = abc_row['商品品类']
                    abc_sales = abc_row['ABC分类（按销售额）']
                    sens_row = price_sensitivity[price_sensitivity['维度值'] == category]
                    if len(sens_row) == 0:
                        continue
                    sens_level = sens_row['敏感度等级'].iloc[0]

                    if abc_sales == 'A类（核心）':
                        if sens_level == '高敏感度（价格主导）':
                            strategy = "保销量：日常定价维持品类均价-5%，大促期间'满减+赠品'"
                            inventory = "高安全库存（月销量1.5倍），提前30天备货"
                        else:
                            strategy = "提利润：高端款溢价10%-15%，常规款维持均价，非大促不降价"
                            inventory = "中等库存（月销量1.2倍），建立区域共享库存池"
                    elif abc_sales == 'B类（潜力）':
                        strategy = "促转化：组合促销，新用户首单折扣5%-8%，提升品类渗透率"
                        inventory = "动态库存（参考预测销量1.1倍），每月调整一次"
                    else:
                        strategy = "清库存：捆绑销售，或限时折扣30%-50%，减少资金占用"
                        inventory = "低库存（月销量0.8倍），滞销超60天直接下架"

                    strategies.append({
                        '策略维度': '商品品类',
                        '维度值': category,
                        'ABC分类': abc_sales,
                        '价格敏感度': sens_level,
                        '定价策略': strategy,
                        '库存策略': inventory,
                        '优先级': '高' if abc_sales == 'A类（核心）' else '中' if abc_sales == 'B类（潜力）' else '低'
                    })

            # 2. 区域策略
            if 'region_abc' in self.results:
                for _, region_row in self.results['region_abc'].iterrows():
                    region = region_row['区域']
                    abc_sales = region_row['ABC分类（按销售额）']

                    if abc_sales == 'A类（核心）':
                        strategy = "重点投入：增加区域专属促销，优化物流时效，提升用户留存"
                        resource = "优先配置仓储资源，增加客服团队"
                    elif abc_sales == 'B类（潜力）':
                        strategy = "渗透拓展：与区域KOL合作推广，开设线下体验点"
                        resource = "适度投入广告预算，测试用户偏好"
                    else:
                        strategy = "低成本覆盖：通过社区团购、下沉渠道触达"
                        resource = "控制成本，复用核心区域资源"

                    strategies.append({
                        '策略维度': '区域',
                        '维度值': region,
                        'ABC分类': abc_sales,
                        '运营策略': strategy,
                        '资源配置': resource,
                        '优先级': '高' if abc_sales == 'A类（核心）' else '中' if abc_sales == 'B类（潜力）' else '低'
                    })

            strategy_df = pd.DataFrame(strategies)
            self.results['operation_strategy'] = strategy_df
            return True
        except Exception as e:
            st.error(f"策略生成错误: {str(e)}")
            return False

    def generate_all_results(self):
        """生成所有优化结果"""
        try:
            abc_success = self.abc_analysis()
            sensitivity_success = self.price_sensitivity_analysis()
            strategy_success = self.generate_operation_strategy() if (abc_success and sensitivity_success) else False

            # 整理结果文件
            result_files = {}
            progress_log = []

            if abc_success:
                result_files['01_ABC分类结果（商品品类+区域）.xlsx'] = pd.ExcelWriter(io.BytesIO())
                with result_files['01_ABC分类结果（商品品类+区域）.xlsx'] as writer:
                    self.results['category_abc'].to_excel(writer, sheet_name='商品品类ABC', index=False)
                    if 'region_abc' in self.results:
                        self.results['region_abc'].to_excel(writer, sheet_name='区域ABC', index=False)
                result_files['01_ABC分类结果（商品品类+区域）.xlsx'] = result_files[
                    '01_ABC分类结果（商品品类+区域）.xlsx'].book
                progress_log.append("ABC分类完成：商品品类按销售额/利润分类，区域按销售额分类")

            if sensitivity_success:
                result_files['02_价格敏感度分析结果（品类+人群）.xlsx'] = self.results['price_sensitivity']
                progress_log.append(
                    f"价格敏感度分析完成：覆盖{len(self.results['price_sensitivity'])}个维度值")

            if strategy_success:
                result_files['03_运营策略推荐.xlsx'] = self.results['operation_strategy']
                progress_log.append(
                    f"运营策略生成完成：{len(self.results['operation_strategy'])}条策略")

            return result_files, progress_log
        except Exception as e:
            return None, [f"优化分析错误: {str(e)}"]


# ============================================================================
# 页面函数
# ============================================================================
def show_project_overview():
    """项目概览页面"""
    st.markdown('<div class="main-header"><i class="bi bi-graph-up-arrow icon-large"></i> 电商销售分析与策略优化系统</div>', unsafe_allow_html=True)

    # 系统功能介绍卡片
    st.markdown("""
    <div class="modern-card">
        <h3><i class="bi bi-info-circle-fill text-primary"></i> 系统功能概述</h3>
        <p class="lead">完整的电商销售分析流程，支持从数据导入到策略优化的全链路分析</p>
    </div>
    """, unsafe_allow_html=True)

    # 功能特性展示
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="modern-card text-center">
            <i class="bi bi-file-earmark-excel icon-large text-success"></i>
            <h5>数据预处理</h5>
            <p>生成<br>6个标准化输出文件</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="modern-card text-center">
            <i class="bi bi-diagram-3 icon-large text-info"></i>
            <h5>多维特征分析</h5>
            <p>品类×区域×利润热力图<br>客户-商品聚类分析</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="modern-card text-center">
            <i class="bi bi-graph-up icon-large text-warning"></i>
            <h5>销售预测</h5>
            <p>ARIMA+XGBoost混合模型<br>预测未来销售趋势</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="modern-card text-center">
            <i class="bi bi-lightbulb icon-large text-danger"></i>
            <h5>运营优化</h5>
            <p>ABC分类、价格敏感度分析<br>可落地的运营策略</p>
        </div>
        """, unsafe_allow_html=True)

    # 系统指标展示
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <i class="bi bi-file-earmark-text icon-large"></i>
            <h6>标准输出文件</h6>
            <h3>6个</h3>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <i class="bi bi-list-task icon-large"></i>
            <h6>分析任务</h6>
            <h3>4个</h3>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <i class="bi bi-database icon-large"></i>
            <h6>支持数据格式</h6>
            <h3>Excel/CSV</h3>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="metric-card">
            <i class="bi bi-graph-up-arrow icon-large"></i>
            <h6>预测模型</h6>
            <h3>ARIMA+XGBoost</h3>
        </div>
        """, unsafe_allow_html=True)

    # 任务状态概览
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="modern-card">
        <h3><i class="bi bi-check2-square icon-large"></i> 任务完成状态</h3>
    </div>
    """, unsafe_allow_html=True)

    # 任务状态卡片布局
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    tasks = [
        ("数据预处理", st.session_state.task1_completed, "bi-file-earmark-excel"),
        ("多维分析", st.session_state.task2_completed, "bi-diagram-3"),
        ("销售预测", st.session_state.task3_completed, "bi-graph-up"),
        ("运营优化", st.session_state.task4_completed, "bi-lightbulb")
    ]

    with status_col1:
        task_name, completed, icon_class = tasks[0]
        status_class = "status-completed" if completed else "status-pending"
        status_text = "已完成" if completed else "待完成"
        status_icon = "check-circle-fill" if completed else "clock-fill"

        st.markdown(f"""
        <div class="task-status {status_class}">
            <i class="bi {icon_class} icon-medium"></i>
            <i class="bi bi-{status_icon} icon-medium"></i>
            <div style="margin-left: 0.5rem;">
                <strong>{task_name}</strong>
                <div style="font-size: 0.9rem; color: #94a3b8;">{status_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with status_col2:
        task_name, completed, icon_class = tasks[1]
        status_class = "status-completed" if completed else "status-pending"
        status_text = "已完成" if completed else "待完成"
        status_icon = "check-circle-fill" if completed else "clock-fill"

        st.markdown(f"""
        <div class="task-status {status_class}">
            <i class="bi {icon_class} icon-medium"></i>
            <i class="bi bi-{status_icon} icon-medium"></i>
            <div style="margin-left: 0.5rem;">
                <strong>{task_name}</strong>
                <div style="font-size: 0.9rem; color: #94a3b8;">{status_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with status_col3:
        task_name, completed, icon_class = tasks[2]
        status_class = "status-completed" if completed else "status-pending"
        status_text = "已完成" if completed else "待完成"
        status_icon = "check-circle-fill" if completed else "clock-fill"

        st.markdown(f"""
        <div class="task-status {status_class}">
            <i class="bi {icon_class} icon-medium"></i>
            <i class="bi bi-{status_icon} icon-medium"></i>
            <div style="margin-left: 0.5rem;">
                <strong>{task_name}</strong>
                <div style="font-size: 0.9rem; color: #94a3b8;">{status_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with status_col4:
        task_name, completed, icon_class = tasks[3]
        status_class = "status-completed" if completed else "status-pending"
        status_text = "已完成" if completed else "待完成"
        status_icon = "check-circle-fill" if completed else "clock-fill"

        st.markdown(f"""
        <div class="task-status {status_class}">
            <i class="bi {icon_class} icon-medium"></i>
            <i class="bi bi-{status_icon} icon-medium"></i>
            <div style="margin-left: 0.5rem;">
                <strong>{task_name}</strong>
                <div style="font-size: 0.9rem; color: #94a3b8;">{status_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def task1_data_preprocessing():
    """任务1：数据预处理页面（按论文要求生成标准化输出文件）"""
    st.markdown('<h2><i class="bi bi-file-earmark-excel icon-large text-success"></i> 任务1: 数据预处理</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <h5><i class="bi bi-info-circle-fill"></i> 输出文件</h5>
        <ol>
            <li>电商 步骤1 缺失值统计结果.xlsx</li>
            <li>电商 步骤2 进货价格处理后数据.xlsx</li>
            <li>电商 步骤3 利润修正后数据.xlsx</li>
            <li>电商 步骤4 异常修正及利润重算后数据.xlsx</li>
            <li>电商 步骤5 MinMax标准化后数据.xlsx</li>
            <li>电商 步骤5 ZScore标准化后数据.xlsx</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # 文件上传组件
    st.markdown("""
    <div class="modern-card">
        <h5><i class="bi bi-cloud-upload icon-medium"></i> 数据文件上传</h5>
        <p class="text-muted">支持Excel (.xlsx) 和CSV格式的数据文件</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("选择数据文件", type=["xlsx", "csv"])

    if uploaded_file is not None:
        # 读取文件
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            # 保存原始数据
            st.session_state.raw_data = df
            st.session_state.current_file = uploaded_file.name

            # 数据清洗：自动处理数值列中的非数值字符
            df_clean = clean_numeric_columns(df)

            st.markdown(f"""
            <div class="success-box">
                <h5><i class="bi bi-check-circle-fill"></i> 文件上传成功</h5>
                <p>共读取 <strong>{len(df):,}</strong> 条记录，<strong>{len(df.columns)}</strong> 个字段</p>
                <p><small>文件名: {uploaded_file.name}</small></p>
            </div>
            """, unsafe_allow_html=True)

            # 数据预览和清洗对比
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<h3><i class="bi bi-eye icon-medium"></i> 数据预览</h3>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="modern-card">
                    <h5><i class="bi bi-file-earmark"></i> 原始数据</h5>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(df.head())

            with col2:
                st.markdown("""
                <div class="modern-card">
                    <h5><i class="bi bi-arrow-clockwise"></i> 清洗后数据</h5>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(df_clean.head())

            # 数据类型信息
            st.markdown('<h3><i class="bi bi-info-square icon-medium"></i> 数据结构信息</h3>', unsafe_allow_html=True)
            dtype_df = pd.DataFrame({
                '字段名': df_clean.columns,
                '数据类型': df_clean.dtypes.astype(str)
            })
            st.dataframe(dtype_df)

            # 执行预处理按钮
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if st.button("开始数据预处理", type="primary", use_container_width=True):
                    with st.spinner("正在执行数据预处理..."):
                        preprocessor = Task1Preprocessor(df_clean)
                        result_files, progress_log, final_data, encoders, column_types = preprocessor.generate_all_results()

                    if result_files:
                        # 保存结果到session state
                        st.session_state.step1_missing_data = result_files['电商 步骤1 缺失值统计结果.xlsx']
                        st.session_state.step2_price_data = result_files['电商 步骤2 进货价格处理后数据.xlsx']
                        st.session_state.step3_profit_data = result_files['电商 步骤3 利润修正后数据.xlsx']
                        st.session_state.step4_abnormal_data = result_files['电商 步骤4 异常修正及利润重算后数据.xlsx']
                        st.session_state.step5_minmax_data = result_files['电商 步骤5 MinMax标准化后数据.xlsx']
                        st.session_state.step5_zscore_data = result_files['电商 步骤5 ZScore标准化后数据.xlsx']
                        st.session_state.processed_data = final_data
                        st.session_state.category_encoder = encoders
                        st.session_state.column_types = column_types
                        st.session_state.task1_completed = True

                        st.markdown("""
                        <div class="success-box">
                            <h4><i class="bi bi-check-circle-fill"></i> 数据预处理完成！</h4>
                            <p>已成功生成6个标准化输出文件</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # 1. 展示预处理步骤结果预览
                        st.markdown('<h3><i class="bi bi-eye icon-medium"></i> 预处理步骤结果预览</h3>', unsafe_allow_html=True)

                        # 步骤1：缺失值统计结果
                        st.markdown("""
                        <div class="modern-card">
                            <h5><i class="bi bi-search"></i> 步骤1：缺失值统计结果</h5>
                        </div>
                        """, unsafe_allow_html=True)
                        st.dataframe(st.session_state.step1_missing_data.head())

                        # 步骤2：进货价格处理后数据
                        st.markdown("""
                        <div class="modern-card">
                            <h5><i class="bi bi-currency-exchange"></i> 步骤2：进货价格处理后数据</h5>
                        </div>
                        """, unsafe_allow_html=True)
                        st.dataframe(st.session_state.step2_price_data[['商品品类', '进货价格']].head())

                        # 步骤3：利润修正后数据
                        st.markdown("""
                        <div class="modern-card">
                            <h5><i class="bi bi-graph-up-arrow"></i> 步骤3：利润修正后数据</h5>
                        </div>
                        """, unsafe_allow_html=True)
                        if '利润是否正确' in st.session_state.step3_profit_data.columns:
                            st.dataframe(
                                st.session_state.step3_profit_data[['商品品类', '利润', '利润是否正确']].head())
                        else:
                            st.dataframe(st.session_state.step3_profit_data[['商品品类', '利润']].head())

                        # 2. 展示预处理日志
                        st.markdown('<h3><i class="bi bi-terminal icon-medium"></i> 预处理执行日志</h3>', unsafe_allow_html=True)
                        for log in progress_log:
                            st.markdown(f"""
                            <div class="info-box">
                                <i class="bi bi-check-circle"></i> {log}
                            </div>
                            """, unsafe_allow_html=True)

                        # 3. 提供结果文件下载（按要求的文件名）
                        st.markdown('<h3><i class="bi bi-download icon-medium"></i> 要求输出文件下载</h3>', unsafe_allow_html=True)

                        # 文件下载网格布局
                        files_grid = [
                            ('电商 步骤1 缺失值统计结果.xlsx', 'bi-search', 'primary'),
                            ('电商 步骤2 进货价格处理后数据.xlsx', 'bi-currency-exchange', 'success'),
                            ('电商 步骤3 利润修正后数据.xlsx', 'bi-graph-up-arrow', 'warning'),
                            ('电商 步骤4 异常修正及利润重算后数据.xlsx', 'bi-exclamation-triangle', 'info'),
                            ('电商 步骤5 MinMax标准化后数据.xlsx', 'bi-bar-chart', 'primary'),
                            ('电商 步骤5 ZScore标准化后数据.xlsx', 'bi-bar-chart-fill', 'success')
                        ]

                        col1, col2, col3 = st.columns(3)
                        for i, (filename, icon, color) in enumerate(files_grid):
                            col = [col1, col2, col3][i % 3]
                            data = result_files[filename]

                            with col:
                                download_data = None  # 初始化变量
                                if isinstance(data, pd.ExcelWriter):
                                    excel_bytes = io.BytesIO()
                                    data.save(excel_bytes)
                                    excel_bytes.seek(0)
                                    download_data = excel_bytes
                                elif isinstance(data, pd.DataFrame):
                                    excel_bytes = io.BytesIO()
                                    with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
                                        data.to_excel(writer, index=False)
                                    download_data = excel_bytes.getvalue()

                                st.markdown(f"""
                                <div class="modern-card text-center">
                                    <i class="bi {icon} icon-large text-{color}"></i>
                                    <h6>{filename}</h6>
                                </div>
                                """, unsafe_allow_html=True)

                                if download_data is not None:
                                    st.download_button(
                                        label=f"下载",
                                        data=download_data,
                                        file_name=filename,
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True
                                    )

                        # 提示下一步操作
                        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                        st.markdown("""
                        <div class="warning-box">
                            <h5><i class="bi bi-arrow-right-circle"></i> 下一步操作</h5>
                            <p>数据预处理已完成！现在可以继续进行 <strong>多维销售特征分析（任务2）</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="error-box">
                            <h5><i class="bi bi-x-circle-fill"></i> 预处理失败</h5>
                            <p>请检查数据格式或查看错误日志</p>
                        </div>
                        """, unsafe_allow_html=True)
                        for log in progress_log:
                            st.markdown(f"""
                            <div class="error-box">
                                <i class="bi bi-x-circle"></i> {log}
                            </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <h5><i class="bi bi-x-circle-fill"></i> 文件读取错误</h5>
                <p><code>{str(e)}</code></p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            <h5><i class="bi bi-info-circle-fill"></i> 准备开始数据预处理</h5>
            <p>请上传原始数据文件开始预处理流程</p>
            <p><strong>建议包含字段：</strong>商品品类、区域、销售额、利润、日期等</p>
        </div>
        """, unsafe_allow_html=True)


def show_feature_analysis_info(show_action_prompt=True):
    """显示多维特征分析功能说明"""
    st.markdown("### 📊 多维特征分析功能说明")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; padding: 20px; border-radius: 15px; height: 100%;">
            <h6>📈 Python可视化展示模式</h6>
            <ul style="color: white; margin-left: 20px;">
                <li>交叉维度热力图分析</li>
                <li>客户-商品聚类分析</li>
                <li>系统内置可视化图表</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white; padding: 20px; border-radius: 15px; height: 100%;">
            <h6>📋 图表数据导出模式</h6>
            <ul style="color: white; margin-left: 20px;">
                <li>城市分布数据</li>
                <li>省份分布数据</li>
                <li>城市分级数据</li>
                <li>区域分级数据</li>
                <li>性别-品类数据</li>
                <li>年龄-性别数据</li>
                <li>时间序列数据</li>
                <li>相关性矩阵数据</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if show_action_prompt:
        st.markdown("---")
        st.markdown("### 🚀 开始分析")
        st.info("**请先完成数据预处理，然后选择分析模式开始多维特征分析！**")


def enhanced_task2_multidimensional_analysis():
    """增强版多维分析页面 - 修复对象序列化问题"""
    st.markdown('<h2><i class="bi bi-diagram-3 icon-large text-info"></i> 任务2: 多维销售特征分析</h2>',
                unsafe_allow_html=True)

    # 如果数据预处理未完成，显示功能说明和警告
    if not st.session_state.task1_completed:
        show_feature_analysis_info(show_action_prompt=True)
        st.warning("⚠️ **需要先完成数据预处理**")
        st.info("请先完成 **数据预处理（任务1）** 才能进行多维分析")
        return

    df = st.session_state.processed_data
    column_types = st.session_state.column_types

    # 检查是否已有分析结果
    has_existing_results = st.session_state.get('task2_completed', False) and st.session_state.get(
        'task2_analysis_data')

    if has_existing_results:
        st.markdown("""
        <div class="success-box">
            <h4><i class="bi bi-check-circle-fill"></i> 多维特征分析已完成！</h4>
            <p>基于已有的分析结果，您可以选择查看方式</p>
        </div>
        """, unsafe_allow_html=True)

        # 显示已有的分析结果
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Python可视化展示", use_container_width=True):
                st.session_state.analysis_mode = "Python可视化展示"
        with col2:
            if st.button("图表数据导出", use_container_width=True):
                st.session_state.analysis_mode = "图表数据导出"

        # 根据选择的模式显示结果
        if st.session_state.get('analysis_mode'):
            if st.session_state.analysis_mode == "Python可视化展示":
                # 使用修复后的函数
                show_python_visualizations_from_dict(st.session_state.task2_results)
            else:
                show_data_export_interface(st.session_state.task2_analysis_data)

        # 重新分析按钮
        if st.button("重新执行多维分析", type="secondary", use_container_width=True):
            st.session_state.task2_completed = False
            st.session_state.analysis_mode = None
            if 'task2_results' in st.session_state:
                del st.session_state.task2_results
            if 'task2_analysis_data' in st.session_state:
                del st.session_state.task2_analysis_data
            st.rerun()

    else:
        # 分析模式选择
        st.markdown('<h3><i class="bi bi-sliders"></i> 选择分析模式</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            python_mode = st.button("Python可视化展示", use_container_width=True,
                                    help="使用内置可视化图表进行分析")
        with col2:
            export_mode = st.button("图表数据导出", use_container_width=True,
                                    help="导出数据用于外部图表制作")

        # 设置分析模式
        if python_mode:
            st.session_state.analysis_mode = "Python可视化展示"
        elif export_mode:
            st.session_state.analysis_mode = "图表数据导出"

        # 执行分析
        if st.session_state.get('analysis_mode'):
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            if st.button("执行多维特征分析", type="primary", use_container_width=True):
                with st.spinner("正在执行多维分析..."):
                    analyzer = EnhancedTask2Analyzer(df, column_types)

                    # 执行基础分析（热力图和聚类）
                    heatmap_success = analyzer.create_heatmaps()
                    cluster_success = analyzer.perform_clustering_analysis()

                    # 生成所有分析数据
                    all_analysis_data = analyzer.generate_all_analysis_data()

                    # 保存结果到session state - 保存字典而不是对象
                    st.session_state.task2_results = analyzer.results
                    st.session_state.task2_analysis_data = all_analysis_data
                    st.session_state.task2_completed = True

                st.markdown("""
                <div class="success-box">
                    <h4><i class="bi bi-check-circle-fill"></i> 多维特征分析完成！</h4>
                    <p>已成功完成多维度销售特征分析</p>
                </div>
                """, unsafe_allow_html=True)

                # 立即显示结果
                if st.session_state.analysis_mode == "Python可视化展示":
                    show_python_visualizations(analyzer)  # 第一次使用对象
                else:
                    show_data_export_interface(all_analysis_data)

        else:
            # 显示功能说明
            show_feature_analysis_info(show_action_prompt=False)
            st.markdown("---")
            st.markdown("### 🚀 选择分析模式")
            st.info("👆 **点击上方按钮开始分析！**")

def display_task3_results(results):
    """显示销售预测的分析结果"""
    # 1. 预测结果可视化
    if 'visualizations' in results:
        st.markdown('<h3><i class="bi bi-graph-up icon-medium"></i> 预测结果可视化</h3>', unsafe_allow_html=True)

        viz_results = results['visualizations']

        # 利润预测对比图（必须展示）
        st.markdown("""
        <div class="modern-card">
            <h5><i class="bi bi-graph-up-arrow"></i> 利润预测对比图</h5>
            <p><strong>图表说明：</strong></p>
            <ul>
                <li>蓝色线：训练集实际利润值</li>
                <li>红色线：测试集实际利润值</li>
                <li>粉色虚线：ARIMA模型预测值</li>
                <li>绿色线：ARIMA+XGBoost最终预测值</li>
                <li>灰色虚线：训练集/测试集分界线</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(viz_results['main_forecast'])

        # 误差分析图
        st.markdown("""
        <div class="modern-card">
            <h5><i class="bi bi-speedometer2"></i> 预测误差分析图</h5>
            <p><strong>图表说明：</strong>显示每日预测的相对误差百分比，MAPE（平均绝对百分比误差）是主要评估指标，误差越小，模型预测精度越高</p>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(viz_results['error_analysis'])

        # 残差分析图
        if 'residual_analysis' in viz_results:
            st.markdown("""
            <div class="modern-card">
                <h5><i class="bi bi-graph-down"></i> 残差分析图</h5>
                <p><strong>图表说明：</strong>显示XGBoost学习到的残差模式，帮助理解模型如何改进ARIMA的预测结果</p>
            </div>
            """, unsafe_allow_html=True)
            st.pyplot(viz_results['residual_analysis'])

    # 2. 模型评估结果
    if 'model_evaluation' in results:
        st.markdown('<h3><i class="bi bi-clipboard-data icon-medium"></i> 模型评估结果</h3>', unsafe_allow_html=True)

        eval_results = results['model_evaluation']
        if isinstance(eval_results, dict):
            col1, col2 = st.columns(2)
            with col1:
                if 'accuracy_metrics' in eval_results:
                    st.markdown('<h6>预测精度指标</h6>', unsafe_allow_html=True)
                    st.dataframe(eval_results['accuracy_metrics'])

            with col2:
                if 'model_comparison' in eval_results:
                    st.markdown('<h6>模型对比分析</h6>', unsafe_allow_html=True)
                    st.dataframe(eval_results['model_comparison'])

    # 3. 预测数据表格
    if 'forecast_data' in results:
        st.markdown('<h3><i class="bi bi-table icon-medium"></i> 预测数据详情</h3>', unsafe_allow_html=True)

        forecast_data = results['forecast_data']
        if isinstance(forecast_data, pd.DataFrame):
            st.dataframe(forecast_data.head(20))

def task3_sales_forecast():
    """任务3：销售预测页面"""
    st.markdown('<h2><i class="bi bi-graph-up icon-large text-warning"></i> 任务3: 销售预测</h2>', unsafe_allow_html=True)

    if not st.session_state.get('task1_completed', False):
        st.markdown("""
        <div class="warning-box">
            <h5><i class="bi bi-exclamation-triangle-fill"></i> 需要先完成数据预处理</h5>
            <p>请先完成<strong>数据预处理（任务1）</strong>才能进行销售预测</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # 获取预处理后的数据
    df = st.session_state.processed_data
    column_types = st.session_state.column_types

    # 预测模型介绍
    st.markdown("""
    <div class="info-box">
        <h5><i class="bi bi-info-circle-fill"></i> ARIMA-XGBoost混合预测模型</h5>
        <div class="row">
            <div class="col-md-6">
                <h6><i class="bi bi-gear"></i> 模型组成</h6>
                <ul>
                    <li><strong>ARIMA(2,1,2)</strong>: 捕捉时间序列趋势</li>
                    <li><strong>XGBoost</strong>: 学习ARIMA的残差模式</li>
                    <li><strong>最终预测</strong>: ARIMA预测 + XGBoost残差预测</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h6><i class="bi bi-calendar-range"></i> 预测设置</h6>
                <ul>
                    <li><strong>训练集</strong>: 11月1-24日（24天数据）</li>
                    <li><strong>测试集</strong>: 11月25-30日（6天数据）</li>
                    <li><strong>评估指标</strong>: MAPE（平均绝对百分比误差）</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 执行预测按钮
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # 检查是否已经完成销售预测
    if st.session_state.task3_completed and hasattr(st.session_state, 'task3_results'):
        st.markdown("""
        <div class="success-box">
            <h4><i class="bi bi-check-circle-fill"></i> 销售预测已完成！</h4>
            <p>基于已有的ARIMA-XGBoost混合预测结果，您可以查看预测结果和下载文件，或重新执行预测</p>
        </div>
        """, unsafe_allow_html=True)

        # 显示已有的预测结果
        display_task3_results(st.session_state.task3_results)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("重新执行ARIMA-XGBoost混合预测", use_container_width=True):
                st.session_state.task3_completed = False
                if hasattr(st.session_state, 'task3_results'):
                    delattr(st.session_state, 'task3_results')
                st.rerun()
    else:
        if st.button("执行ARIMA-XGBoost混合预测", type="primary", use_container_width=True):
            with st.spinner("预测中...（使用ARIMA(2,1,2)+XGBoost混合模型）"):
                forecaster = Task3Forecaster(df, column_types)
                result_files, progress_log = forecaster.generate_all_results()

            if result_files:
                st.session_state.task3_results = forecaster.results
                st.session_state.task3_completed = True

                st.markdown("""
                <div class="success-box">
                    <h4><i class="bi bi-check-circle-fill"></i> 销售预测完成！</h4>
                    <p>ARIMA-XGBoost混合模型预测已完成</p>
                </div>
                """, unsafe_allow_html=True)

                # 1. 展示预测结果可视化
                st.markdown('<h3><i class="bi bi-graph-up icon-medium"></i> 预测结果可视化</h3>', unsafe_allow_html=True)

                if 'visualizations' in forecaster.results:
                    viz_results = forecaster.results['visualizations']

                    # 利润预测对比图（必须展示）
                    st.markdown("""
                    <div class="modern-card">
                        <h5><i class="bi bi-graph-up-arrow"></i> 利润预测对比图</h5>
                        <p><strong>图表说明：</strong></p>
                        <ul>
                            <li>蓝色线：训练集实际利润值</li>
                            <li>红色线：测试集实际利润值</li>
                            <li>粉色虚线：ARIMA模型预测值</li>
                            <li>绿色线：ARIMA+XGBoost最终预测值</li>
                            <li>灰色虚线：训练集/测试集分界线</li>
                        </ul>
                        </div>
                    """, unsafe_allow_html=True)
                    st.pyplot(viz_results['main_forecast'])

                    # 误差分析图
                    st.markdown("""
                    <div class="modern-card">
                        <h5><i class="bi bi-speedometer2"></i> 预测误差分析图</h5>
                        <p><strong>图表说明：</strong>显示每日预测的相对误差百分比，MAPE（平均绝对百分比误差）是主要评估指标，误差越小，模型预测精度越高</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.pyplot(viz_results['error_analysis'])

                    # 残差分析图
                    st.markdown("""
                    <div class="modern-card">
                        <h5><i class="bi bi-activity"></i> ARIMA模型残差分析图</h5>
                        <p><strong>图表说明：</strong>显示ARIMA模型在训练集上的残差分布，残差越接近0且波动越小，说明ARIMA模型拟合越好，为XGBoost提供学习目标</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.pyplot(viz_results['residual_analysis'])

                    # 特征重要性排名图
                    st.markdown("""
                    <div class="modern-card">
                        <h5><i class="bi bi-bar-chart"></i> 特征重要性排名图</h5>
                        <p><strong>图表说明：</strong>显示XGBoost模型中各特征的重要性得分，重要性越高，该特征对残差预测的贡献越大，帮助理解模型决策依据</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.pyplot(viz_results['feature_importance'])

                    # 2. 展示预测结果表格
                    st.markdown('<h3><i class="bi bi-table icon-medium"></i> 预测结果详情</h3>', unsafe_allow_html=True)
                    if 'detailed_results' in forecaster.results:
                        forecast_df = forecaster.results['detailed_results']
                        st.dataframe(forecast_df.round(2))

                        # 关键指标总结
                        st.markdown('<h4><i class="bi bi-speedometer icon-medium"></i> 预测关键指标</h4>', unsafe_allow_html=True)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            mape = forecaster.results.get('mape', 0)
                            st.markdown(f"""
                            <div class="metric-card">
                                <i class="bi bi-percent icon-large"></i>
                                <h6>测试集MAPE</h6>
                                <h3>{mape:.2f}%</h3>
                            </div>
                            """, unsafe_allow_html=True)

                        with col2:
                            best_error = forecast_df['相对误差(%)'].min()
                            best_day = forecast_df.loc[forecast_df['相对误差(%)'].idxmin(), '日期']
                            st.markdown(f"""
                            <div class="metric-card">
                            <i class="bi bi-award icon-large"></i>
                            <h6>最佳预测精度</h6>
                            <h3>{best_error:.1f}%</h3>
                            <p>11月{int(best_day)}日</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            worst_error = forecast_df['相对误差(%)'].max()
                            worst_day = forecast_df.loc[forecast_df['相对误差(%)'].idxmax(), '日期']
                            st.markdown(f"""
                            <div class="metric-card">
                                <i class="bi bi-exclamation-triangle icon-large"></i>
                                <h6>最差预测精度</h6>
                                <h3>{worst_error:.1f}%</h3>
                                <p>11月{int(worst_day)}日</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # 3. 特征重要性分析
                    st.markdown('<h3><i class="bi bi-list-ul icon-medium"></i> 特征重要性分析</h3>', unsafe_allow_html=True)
                    if 'feature_importance' in forecaster.results:
                        feature_importance = forecaster.results['feature_importance']
                        st.dataframe(feature_importance.round(4))

                        st.markdown("""
                        <div class="info-box">
                            <h6><i class="bi bi-lightbulb"></i> 特征重要性解读</h6>
                            <ul>
                                <li><strong>高重要性特征</strong>：对模型预测影响最大的因素</li>
                                <li><strong>中等重要性特征</strong>：有一定预测价值的辅助因素</li>
                                <li><strong>低重要性特征</strong>：对预测结果影响较小</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    # 4. 进度日志
                    st.markdown('<h3><i class="bi bi-terminal icon-medium"></i> 预测执行日志</h3>', unsafe_allow_html=True)
                    for log in progress_log:
                        st.markdown(f"""
                        <div class="info-box">
                            <i class="bi bi-check-circle"></i> {log}
                        </div>
                        """, unsafe_allow_html=True)

                    # 5. 文件下载
                    st.markdown('<h3><i class="bi bi-download icon-medium"></i> 预测结果文件下载</h3>', unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    for i, (filename, data) in enumerate(result_files.items()):
                        col = [col1, col2, col3][i % 3]
                        with col:
                            if isinstance(data, pd.DataFrame):
                                excel_bytes = io.BytesIO()
                                with pd.ExcelWriter(excel_bytes, engine='openpyxl') as writer:
                                    data.to_excel(writer, index=False)

                            st.markdown(f"""
                                <div class="modern-card text-center">
                                    <i class="bi bi-file-earmark-excel icon-large text-success"></i>
                                    <h6>{filename}</h6>
                                </div>
                                """, unsafe_allow_html=True)

                            st.download_button(
                                label="下载",
                                data=excel_bytes.getvalue(),
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

                    # 提示下一步操作
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    st.markdown("""
                    <div class="warning-box">
                        <h5><i class="bi bi-arrow-right-circle"></i> 下一步操作</h5>
                        <p>销售预测已完成！现在可以继续进行 <strong>运营策略优化（任务4）</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="error-box">
                        <h5><i class="bi bi-x-circle-fill"></i> 预测失败</h5>
                        <p>请查看错误日志</p>
                    </div>
                    """, unsafe_allow_html=True)
                    for log in progress_log:
                        st.markdown(f"""
                        <div class="error-box">
                            <i class="bi bi-x-circle"></i> {log}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <h5><i class="bi bi-info-circle-fill"></i> 预测模型说明</h5>
                <ul>
                    <li>使用ARIMA模型捕捉时间序列趋势</li>
                    <li>使用XGBoost模型学习ARIMA的残差模式</li>
                    <li>最终预测 = ARIMA预测 + XGBoost残差预测</li>
                    <li>测试集：11月25-30日</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def display_task4_results(results):
    """显示运营优化的分析结果"""
    # 1. ABC分类结果展示
    if 'abc_analysis' in results:
        st.markdown('<h3><i class="bi bi-sort-numeric-down icon-medium"></i> ABC分类分析结果</h3>', unsafe_allow_html=True)

        abc_result = results['abc_analysis']
        if isinstance(abc_result, dict):
            # 显示ABC分类结果
            col1, col2 = st.columns(2)
            with col1:
                if 'category_abc' in abc_result:
                    st.markdown('<h6>商品品类ABC分类</h6>', unsafe_allow_html=True)
                    st.dataframe(abc_result['category_abc'])

            with col2:
                if 'region_abc' in abc_result:
                    st.markdown('<h6>区域ABC分类</h6>', unsafe_allow_html=True)
                    st.dataframe(abc_result['region_abc'])

    # 2. 价格敏感度分析结果
    if 'price_sensitivity' in results:
        st.markdown('<h3><i class="bi bi-graph-down icon-medium"></i> 价格敏感度分析结果</h3>', unsafe_allow_html=True)

        price_result = results['price_sensitivity']
        if isinstance(price_result, dict):
            col1, col2 = st.columns(2)
            with col1:
                if 'category_sensitivity' in price_result:
                    st.markdown('<h6>品类价格敏感度</h6>', unsafe_allow_html=True)
                    st.dataframe(price_result['category_sensitivity'])

            with col2:
                if 'customer_sensitivity' in price_result:
                    st.markdown('<h6>客户群体敏感度</h6>', unsafe_allow_html=True)
                    st.dataframe(price_result['customer_sensitivity'])

    # 3. 运营策略建议
    if 'operation_strategy' in results:
        st.markdown('<h3><i class="bi bi-clipboard-check icon-medium"></i> 运营策略建议</h3>', unsafe_allow_html=True)

        strategy = results['operation_strategy']
        if isinstance(strategy, dict):
            # 高优先级策略
            if 'high_priority' in strategy and strategy['high_priority']:
                st.markdown('<h6><i class="bi bi-exclamation-triangle-fill text-danger"></i> 高优先级策略</h6>', unsafe_allow_html=True)
                for i, item in enumerate(strategy['high_priority'], 1):
                    st.markdown(f"""
                    <div class="modern-card">
                        <h6>{i}. {item.get('title', '策略建议')}</h6>
                        <p>{item.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # 中优先级策略
            if 'medium_priority' in strategy and strategy['medium_priority']:
                st.markdown('<h6><i class="bi bi-info-circle-fill text-warning"></i> 中优先级策略</h6>', unsafe_allow_html=True)
                for i, item in enumerate(strategy['medium_priority'], 1):
                    st.markdown(f"""
                    <div class="modern-card">
                        <h6>{i}. {item.get('title', '策略建议')}</h6>
                        <p>{item.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # 低优先级策略
            if 'low_priority' in strategy and strategy['low_priority']:
                st.markdown('<h6><i class="bi bi-check-circle-fill text-success"></i> 低优先级策略</h6>', unsafe_allow_html=True)
                for i, item in enumerate(strategy['low_priority'], 1):
                    st.markdown(f"""
                    <div class="modern-card">
                        <h6>{i}. {item.get('title', '策略建议')}</h6>
                        <p>{item.get('description', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)




def task4_operation_optimization():
    st.header("任务4：运营策略优化分析")

    # 在函数开头定义 df 变量
    df = None

    # 检查是否已完成数据预处理
    if not st.session_state.get('task1_completed', False):
        st.markdown("""
            <div class="warning-box">
                <h5><i class="bi bi-exclamation-triangle-fill"></i> 需要先完成数据预处理</h5>
                <p>请先完成<strong>数据预处理（任务1）</strong>才能进行运营优化分析</p>
            </div>
            """, unsafe_allow_html=True)
        return

    # 使用任务1预处理后的数据
    if st.session_state.step5_minmax_data is not None:
        df = st.session_state.step5_minmax_data
    else:
        st.error("❌ 未找到标准化后的数据，请先完成数据预处理")
        return

    # 检查是否已经完成运营优化分析
    if st.session_state.task4_completed and hasattr(st.session_state, 'task4_results'):
        st.markdown("""
            <div class="success-box">
                <h4><i class="bi bi-check-circle-fill"></i> 运营优化分析已完成！</h4>
                <p>基于已有的分析结果，您可以查看各项优化建议</p>
            </div>
            """, unsafe_allow_html=True)

        # 这里显示已有的分析结果（保持你现有的显示逻辑）
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "ABC商品分类",
            "价格敏感度分析",
            "用户分层分析",
            "场景关联分析",
            "综合策略生成"
        ])

        # -------------------------- ABC商品分类 --------------------------
        with tab1:
            st.subheader("ABC商品分类分析")

            # ABC分类计算
            category_summary = df.groupby('商品品类').agg({
                '销售额': 'sum',
                '利润': 'sum'
            }).reset_index()

            total_sales = category_summary['销售额'].sum()
            total_profit = category_summary['利润'].sum()
            category_summary['销售额占比'] = (category_summary['销售额'] / total_sales).round(4)
            category_summary['利润占比'] = (category_summary['利润'] / total_profit).round(4)

            category_summary = category_summary.sort_values(by='销售额占比', ascending=False)
            category_summary['销售额累计占比'] = category_summary['销售额占比'].cumsum().round(4)

            category_summary['ABC分类'] = np.where(
                category_summary['销售额累计占比'] <= 0.7, 'A',
                np.where(category_summary['销售额累计占比'] <= 0.9, 'B', 'C')
            )

            # 显示ABC分类结果
            col1, col2 = st.columns([2, 1])

            with col1:
                st.dataframe(category_summary.style.format({
                    '销售额': '{:,.2f}',
                    '利润': '{:,.2f}',
                    '销售额占比': '{:.2%}',
                    '利润占比': '{:.2%}'
                }), use_container_width=True)

            with col2:
                # ABC分类统计
                abc_stats = category_summary.groupby('ABC分类').agg({
                    '商品品类': 'count',
                    '销售额占比': 'sum',
                    '利润占比': 'sum'
                }).rename(columns={'商品品类': '品类数量'})
                abc_stats['品类占比'] = (abc_stats['品类数量'] / abc_stats['品类数量'].sum()).round(4)

                st.metric("A类品类数", len(category_summary[category_summary['ABC分类'] == 'A']))
                st.metric("B类品类数", len(category_summary[category_summary['ABC分类'] == 'B']))
                st.metric("C类品类数", len(category_summary[category_summary['ABC分类'] == 'C']))

            # ABC分类可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 柱状图
            color_map = {'A': '#1f77b4', 'B': '#7fbf7b', 'C': '#ff7f0e'}
            bars = ax1.bar(
                category_summary['商品品类'],
                category_summary['销售额占比'],
                color=[color_map[cls] for cls in category_summary['ABC分类']],
                alpha=0.8
            )
            ax1.set_title('ABC分类 - 销售额占比', fontsize=14, pad=20)
            ax1.set_xlabel('商品品类')
            ax1.set_ylabel('销售额占比')
            ax1.tick_params(axis='x', rotation=45)
            plt.sca(ax1)
            plt.xticks(ha='right')
            ax1.grid(axis='y', alpha=0.3)

            # 饼图
            abc_count = category_summary['ABC分类'].value_counts()
            ax2.pie(
                abc_count.values,
                labels=[f'{cls}类 ({abc_count[cls]}个)' for cls in abc_count.index],
                colors=[color_map[cls] for cls in abc_count.index],
                autopct='%1.1f%%',
                startangle=90
            )
            ax2.set_title('ABC分类分布', fontsize=14, pad=20)

            plt.tight_layout()
            st.pyplot(fig)

        # -------------------------- 价格敏感度分析 --------------------------
        with tab2:
            st.subheader("价格敏感度分析")

            # 价格敏感度计算函数
            def calculate_price_sensitivity(df, category_col='商品品类', price_col='实际售价', qty_col='销售数'):
                categories = df[category_col].unique()
                sensitivity_results = []

                for cat in categories:
                    subset = df[df[category_col] == cat]
                    if len(subset) < 10:
                        continue

                    # 等频8分位数分组
                    subset = subset.sort_values(price_col)
                    try:
                        subset['price_bin'] = pd.qcut(subset[price_col], q=8, duplicates='drop')
                    except:
                        continue

                    bin_summary = subset.groupby('price_bin').agg({
                        price_col: 'mean',
                        qty_col: 'sum'
                    }).reset_index()

                    if len(bin_summary) < 3:
                        continue

                    # 计算敏感度系数 S
                    Qk = bin_summary[qty_col]
                    Pk = bin_summary[price_col]
                    Q_bar = Qk.mean()
                    P_bar = Pk.mean()

                    numerator = (Qk * ((Qk - Q_bar) / Q_bar)).sum()
                    denominator = (Qk * ((Pk - P_bar) / P_bar)).sum()

                    if denominator != 0:
                        S = numerator / denominator
                    else:
                        S = np.nan

                    # 判断敏感度等级
                    if S < -0.5:
                        level = "强敏感"
                    elif S < 0:
                        level = "弱敏感"
                    else:
                        level = "不敏感"

                    sensitivity_results.append({
                        '品类': cat,
                        '敏感度系数S': round(S, 4),
                        '敏感度等级': level
                    })

                return pd.DataFrame(sensitivity_results)

            sensitivity_df = calculate_price_sensitivity(df)

            if not sensitivity_df.empty:
                # 显示敏感度结果
                st.dataframe(sensitivity_df, use_container_width=True)

                # 价格-销量拟合图生成
                st.subheader("价格-销量拟合分析")

                def generate_fit_plot(category):
                    cat_data = df[df['商品品类'] == category].sort_values('实际售价').copy()

                    # 等频分箱
                    try:
                        cat_data['price_bin'] = pd.qcut(cat_data['实际售价'], q=8, duplicates='drop')
                    except:
                        return None

                    bin_summary = cat_data.groupby('price_bin').agg({
                        '实际售价': 'mean',
                        '销售数': 'sum'
                    }).dropna().reset_index()

                    if len(bin_summary) < 3:
                        return None

                    P = bin_summary['实际售价'].values
                    Q = bin_summary['销售数'].values

                    # 拟合函数
                    def linear_func(p, k, b):
                        return k * p + b

                    def quadratic_func(p, a, b, c):
                        return a * p ** 2 + b * p + c

                    # 生成图表
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(P, Q, color='#2E86AB', s=80, alpha=0.8, label='实际区间数据')

                    # 线性拟合
                    try:
                        linear_params, _ = curve_fit(linear_func, P, Q, maxfev=1000)
                        p_linear = np.linspace(P.min() * 0.9, P.max() * 1.1, 100)
                        ax.plot(p_linear, linear_func(p_linear, *linear_params),
                                color='#E74C3C', linewidth=2, linestyle='--',
                                label='线性拟合')
                    except:
                        pass

                    # 二次拟合
                    try:
                        quadratic_params, _ = curve_fit(
                            quadratic_func, P, Q,
                            bounds=([-np.inf, -np.inf, 0], [0, np.inf, np.inf]),
                            maxfev=1000
                        )
                        p_quad = np.linspace(P.min() * 0.9, P.max() * 1.1, 100)
                        ax.plot(p_quad, quadratic_func(p_quad, *quadratic_params),
                                color='#27AE60', linewidth=2, label='二次拟合')

                        # 临界点
                        a, b_quad, c = quadratic_params
                        critical_price = -b_quad / (2 * a) if a != 0 else np.nan
                        if P.min() * 0.8 <= critical_price <= P.max() * 1.2:
                            ax.axvline(x=critical_price, color='#F39C12', linestyle='-.',
                                       label=f'临界点: {critical_price:.0f}元')
                    except:
                        pass

                    ax.set_title(f'{category} 价格-销量关系')
                    ax.set_xlabel('实际售价（元）')
                    ax.set_ylabel('销售数（件）')
                    ax.legend()
                    ax.grid(alpha=0.3)

                    return fig

                # 选择品类查看拟合图
                selected_category = st.selectbox("选择品类查看价格-销量拟合图", sensitivity_df['品类'].unique())
                if selected_category:
                    fig = generate_fit_plot(selected_category)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.warning("该品类数据不足，无法生成拟合图")
            else:
                st.warning("没有足够数据计算价格敏感度")

        # -------------------------- 用户分层分析 --------------------------
        with tab3:
            st.subheader("用户分层价格偏好分析")

            def user_price_preference_analysis(df):
                # 计算全品类平均价格（加权）
                P_bar = (df['实际售价'] * df['销售数']).sum() / df['销售数'].sum()

                # 用户分层
                df['用户分层'] = df['客户年龄'].apply(
                    lambda x: '青年' if x < 30 else '中年' if x < 50 else '老年'
                ) + '_' + df['客户性别'].apply(lambda x: '男' if x == '男' else '女')

                # 计算各分层指标
                layers = df['用户分层'].unique()
                results = []

                for layer in layers:
                    subset = df[df['用户分层'] == layer]
                    total_sales_qty = subset['销售数'].sum()

                    if total_sales_qty == 0:
                        continue

                    avg_price = (subset['实际售价'] * subset['销售数']).sum() / total_sales_qty

                    # R1: 价格接受度
                    R1 = avg_price / P_bar

                    # R2: 价格集中度
                    low_price_sales = subset[subset['实际售价'] < P_bar]['销售数'].sum()
                    high_price_sales = subset[subset['实际售价'] >= P_bar]['销售数'].sum()

                    R2_low = low_price_sales / total_sales_qty
                    R2_high = high_price_sales / total_sales_qty

                    # R3: 敏感倾向指数
                    if R2_high != 0:
                        R3 = R2_low / R2_high
                    else:
                        R3 = np.inf

                    # 判断敏感度
                    if R3 > 2.5:
                        sense_level = "强敏感"
                    elif R3 > 1.5:
                        sense_level = "中敏感"
                    else:
                        sense_level = "不敏感"

                    results.append({
                        '用户分层': layer,
                        'R1': round(R1, 4),
                        'R2_low': round(R2_low, 4),
                        'R2_high': round(R2_high, 4),
                        'R3': round(R3, 4),
                        '敏感度等级': sense_level
                    })

                return pd.DataFrame(results)

            user_pref_df = user_price_preference_analysis(df)

            if not user_pref_df.empty:
                # 显示用户分层结果
                st.dataframe(user_pref_df, use_container_width=True)

                # 可视化
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # R1可视化
                user_pref_sorted = user_pref_df.sort_values('R1', ascending=False)
                bars1 = ax1.bar(user_pref_sorted['用户分层'], user_pref_sorted['R1'],
                                color='skyblue', alpha=0.8)
                ax1.axhline(y=1, color='red', linestyle='--', label='平均价格线')
                ax1.set_title('用户分层价格接受度 (R1)')
                ax1.set_ylabel('R1 (价格接受度)')
                ax1.tick_params(axis='x', rotation=45)
                ax1.legend()

                # R3可视化
                user_pref_sorted = user_pref_df.sort_values('R3', ascending=False)
                colors = ['red' if x == '强敏感' else 'orange' if x == '中敏感' else 'green'
                          for x in user_pref_sorted['敏感度等级']]
                bars2 = ax2.bar(user_pref_sorted['用户分层'], user_pref_sorted['R3'],
                                color=colors, alpha=0.8)
                ax2.axhline(y=1.5, color='orange', linestyle='--', label='中敏感阈值')
                ax2.axhline(y=2.5, color='red', linestyle='--', label='强敏感阈值')
                ax2.set_title('用户分层价格敏感度 (R3)')
                ax2.set_ylabel('R3 (敏感倾向指数)')
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend()

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("用户分层分析数据不足")

        # -------------------------- 场景关联分析 --------------------------
        with tab4:
            st.subheader("场景关联分析")

            # 场景分析配置
            st.write("### 营销场景价格敏感度分析")

            # 创建场景定义
            scenarios = {
                '促销期': '分析促销活动期间的价格敏感度变化',
                '节假日': '节假日特殊消费场景分析',
                '新品上市': '新品推广期的价格策略效果',
                '常规期': '日常销售的价格敏感度基准'
            }

            selected_scenario = st.selectbox("选择分析场景", list(scenarios.keys()))
            st.info(f"当前场景：{selected_scenario} - {scenarios[selected_scenario]}")

            # 场景模拟分析
            col1, col2 = st.columns(2)

            with col1:
                st.write("#### 场景特征分析")

                # 基于场景的敏感度模拟
                scenario_multipliers = {
                    '促销期': 1.8,  # 促销期敏感度提高
                    '节假日': 1.3,  # 节假日适度提高
                    '新品上市': 0.7,  # 新品期敏感度降低
                    '常规期': 1.0  # 基准
                }

                multiplier = scenario_multipliers[selected_scenario]

                # 计算场景调整后的敏感度
                if 'sensitivity_df' in locals() and not sensitivity_df.empty:
                    scenario_sensitivity = sensitivity_df.copy()
                    scenario_sensitivity['场景调整系数'] = multiplier
                    scenario_sensitivity['场景敏感度'] = scenario_sensitivity['敏感度系数S'] * multiplier

                    # 重新分类场景敏感度
                    def classify_scenario_sensitivity(s_value):
                        if s_value < -0.8:
                            return "极高敏感"
                        elif s_value < -0.4:
                            return "高敏感"
                        elif s_value < 0:
                            return "中等敏感"
                        else:
                            return "低敏感"

                    scenario_sensitivity['场景敏感等级'] = scenario_sensitivity['场景敏感度'].apply(
                        classify_scenario_sensitivity
                    )

                    st.dataframe(
                        scenario_sensitivity[['品类', '敏感度系数S', '场景调整系数', '场景敏感度', '场景敏感等级']],
                        use_container_width=True)

            with col2:
                st.write("#### 场景策略建议")

                scenario_strategies = {
                    '促销期': [
                        "🔥 重点品类加大折扣力度",
                        "🎯 设置阶梯式满减优惠",
                        "⚡ 限时抢购提升转化率",
                        "📱 推送个性化优惠券"
                    ],
                    '节假日': [
                        "🎁 节日主题营销活动",
                        "👨‍👩‍👧‍👦 家庭套装组合销售",
                        "🎄 限量版节日特别款",
                        "💝 礼品包装增值服务"
                    ],
                    '新品上市': [
                        "🌟 首发优惠吸引尝鲜",
                        "📢 KOL合作推广造势",
                        "🎪 线下体验活动联动",
                        "🔍 用户反馈快速迭代"
                    ],
                    '常规期': [
                        "📊 数据监控价格带表现",
                        "🔄 A/B测试优化定价",
                        "🎯 精准用户群体营销",
                        "📈 会员体系深度运营"
                    ]
                }

                strategies = scenario_strategies[selected_scenario]
                for i, strategy in enumerate(strategies, 1):
                    st.write(f"{i}. {strategy}")

            # 场景对比分析
            st.write("### 多场景对比分析")

            # 创建场景对比数据
            scenario_comparison = []
            for scenario, mult in scenario_multipliers.items():
                if 'sensitivity_df' in locals() and not sensitivity_df.empty:
                    avg_sensitivity = sensitivity_df['敏感度系数S'].mean() * mult
                    scenario_comparison.append({
                        '场景': scenario,
                        '调整系数': mult,
                        '平均敏感度': round(avg_sensitivity, 4),
                        '描述': scenarios[scenario]
                    })

            scenario_df = pd.DataFrame(scenario_comparison)

            # 显示场景对比
            col1, col2 = st.columns([1, 2])

            with col1:
                st.dataframe(scenario_df, use_container_width=True)

            with col2:
                # 场景对比可视化
                fig, ax = plt.subplots(figsize=(10, 6))
                scenarios_plot = scenario_df.sort_values('平均敏感度')
                colors = ['#ff9999' if x < 0 else '#66b3ff' for x in scenarios_plot['平均敏感度']]
                bars = ax.barh(scenarios_plot['场景'], scenarios_plot['平均敏感度'], color=colors, alpha=0.8)

                # 添加数值标签
                for bar, value in zip(bars, scenarios_plot['平均敏感度']):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                            f'{value:.3f}', va='center', fontsize=10)

                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='敏感度零线')
                ax.set_xlabel('平均价格敏感度')
                ax.set_title('不同营销场景的价格敏感度对比')
                ax.legend()
                ax.grid(axis='x', alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

        # -------------------------- 综合策略生成 --------------------------
        with tab5:
            st.subheader("综合运营策略生成")

            # 确保有ABC分类和敏感度数据
            if 'category_summary' in locals() and 'sensitivity_df' in locals() and not sensitivity_df.empty:

                # 综合策略生成
                def generate_integrated_strategies(abc_df, sensitivity_df):
                    strategies = []

                    for _, abc_row in abc_df.iterrows():
                        cat = abc_row['商品品类']
                        abc_class = abc_row['ABC分类']

                        # 获取敏感度信息
                        sens_info = sensitivity_df[sensitivity_df['品类'] == cat]
                        if not sens_info.empty:
                            sens_level = sens_info['敏感度等级'].iloc[0]
                            S_value = sens_info['敏感度系数S'].iloc[0]
                        else:
                            sens_level = "未知"
                            S_value = 0

                        # 基于ABC分类和敏感度生成策略
                        if abc_class == 'A':
                            if sens_level == "强敏感":
                                strategy = "🔥 核心高敏感品类：重点监控价格，采用动态定价，推送高折扣券，设置价格预警"
                            elif sens_level == "弱敏感":
                                strategy = "⭐ 核心低敏感品类：适度溢价，强调品质和服务，推会员专属权益"
                            else:
                                strategy = "💎 核心品类：平衡定价，注重品牌建设，优化产品组合"

                        elif abc_class == 'B':
                            if sens_level == "强敏感":
                                strategy = "📊 重要高敏感品类：竞争性定价，参与平台促销，捆绑销售提升客单价"
                            else:
                                strategy = "📈 重要品类：适中定价，买赠活动，交叉销售推荐"

                        else:  # C类
                            if sens_level == "强敏感":
                                strategy = "🔄 一般高敏感品类：清理库存为主，大幅折扣，限时抢购"
                            else:
                                strategy = "📦 一般品类：维持现状，减少营销投入，自然销售"

                        strategies.append({
                            '品类': cat,
                            'ABC分类': abc_class,
                            '价格敏感度': sens_level,
                            '敏感度系数': S_value,
                            '推荐策略': strategy
                        })

                    return pd.DataFrame(strategies)

                strategy_df = generate_integrated_strategies(category_summary, sensitivity_df)

                # 显示策略表
                st.dataframe(strategy_df, use_container_width=True)

                # 策略统计
                st.subheader("策略分布统计")
                col1, col2, col3 = st.columns(3)

                with col1:
                    high_sens = len(strategy_df[strategy_df['价格敏感度'] == '强敏感'])
                    st.metric("强敏感品类", high_sens)

                with col2:
                    medium_sens = len(strategy_df[strategy_df['价格敏感度'] == '弱敏感'])
                    st.metric("弱敏感品类", medium_sens)

                with col3:
                    low_sens = len(strategy_df[strategy_df['价格敏感度'] == '不敏感'])
                    st.metric("不敏感品类", low_sens)

                # 策略建议总结
                st.subheader("关键策略建议")

                # A类品类策略
                a_strategies = strategy_df[strategy_df['ABC分类'] == 'A']
                if not a_strategies.empty:
                    st.write("### 🎯 A类核心品类策略")
                    for _, row in a_strategies.iterrows():
                        st.write(f"- **{row['品类']}** ({row['价格敏感度']})：{row['推荐策略']}")

                # 高敏感品类策略
                high_sens_strategies = strategy_df[strategy_df['价格敏感度'] == '强敏感']
                if not high_sens_strategies.empty:
                    st.write("### 💰 高敏感品类价格策略")
                    for _, row in high_sens_strategies.iterrows():
                        st.write(f"- **{row['品类']}**：重点关注价格竞争力，建议定期调价")

            else:
                st.warning("请先完成ABC分类和价格敏感度分析")

        # ...
    else:
        # 显示开始分析按钮
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("开始运营优化分析", type="primary", use_container_width=True):
                # 设置任务完成状态
                st.session_state.task4_completed = True
                st.rerun()



        # 如果还没有开始分析，显示功能介绍
        if not st.session_state.task4_completed:
            st.markdown("""
            <div class="info-box">
                <h5><i class="bi bi-info-circle-fill"></i> 运营优化分析功能</h5>
                <ul>
                    <li><strong>ABC商品分类</strong>：基于销售额和利润的核心品类识别</li>
                    <li><strong>价格敏感度分析</strong>：品类和用户群体的价格弹性分析</li>
                    <li><strong>用户分层分析</strong>：不同用户群体的消费特征分析</li>
                    <li><strong>场景关联分析</strong>：不同营销场景下的策略建议</li>
                    <li><strong>综合策略生成</strong>：基于多维度分析的运营优化建议</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            return

    # 只有在开始分析后才显示标签页内容

    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "ABC商品分类",
        "价格敏感度分析",
        "用户分层分析",
        "场景关联分析",
        "综合策略生成"
    ])

    # -------------------------- ABC商品分类 --------------------------
    with tab1:
        st.subheader("ABC商品分类分析")

        # ABC分类计算
        category_summary = df.groupby('商品品类').agg({
            '销售额': 'sum',
            '利润': 'sum'
        }).reset_index()

        total_sales = category_summary['销售额'].sum()
        total_profit = category_summary['利润'].sum()
        category_summary['销售额占比'] = (category_summary['销售额'] / total_sales).round(4)
        category_summary['利润占比'] = (category_summary['利润'] / total_profit).round(4)

        category_summary = category_summary.sort_values(by='销售额占比', ascending=False)
        category_summary['销售额累计占比'] = category_summary['销售额占比'].cumsum().round(4)

        category_summary['ABC分类'] = np.where(
            category_summary['销售额累计占比'] <= 0.7, 'A',
            np.where(category_summary['销售额累计占比'] <= 0.9, 'B', 'C')
        )

        # 显示ABC分类结果
        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(category_summary.style.format({
                '销售额': '{:,.2f}',
                '利润': '{:,.2f}',
                '销售额占比': '{:.2%}',
                '利润占比': '{:.2%}'
            }), use_container_width=True)

        with col2:
            # ABC分类统计
            abc_stats = category_summary.groupby('ABC分类').agg({
                '商品品类': 'count',
                '销售额占比': 'sum',
                '利润占比': 'sum'
            }).rename(columns={'商品品类': '品类数量'})
            abc_stats['品类占比'] = (abc_stats['品类数量'] / abc_stats['品类数量'].sum()).round(4)

            st.metric("A类品类数", len(category_summary[category_summary['ABC分类'] == 'A']))
            st.metric("B类品类数", len(category_summary[category_summary['ABC分类'] == 'B']))
            st.metric("C类品类数", len(category_summary[category_summary['ABC分类'] == 'C']))

        # ABC分类可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 柱状图
        color_map = {'A': '#1f77b4', 'B': '#7fbf7b', 'C': '#ff7f0e'}
        bars = ax1.bar(
            category_summary['商品品类'],
            category_summary['销售额占比'],
            color=[color_map[cls] for cls in category_summary['ABC分类']],
            alpha=0.8
        )
        ax1.set_title('ABC分类 - 销售额占比', fontsize=14, pad=20)
        ax1.set_xlabel('商品品类')
        ax1.set_ylabel('销售额占比')
        ax1.tick_params(axis='x', rotation=45)
        plt.sca(ax1)
        plt.xticks(ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # 饼图
        abc_count = category_summary['ABC分类'].value_counts()
        ax2.pie(
            abc_count.values,
            labels=[f'{cls}类 ({abc_count[cls]}个)' for cls in abc_count.index],
            colors=[color_map[cls] for cls in abc_count.index],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('ABC分类分布', fontsize=14, pad=20)

        plt.tight_layout()
        st.pyplot(fig)

    # -------------------------- 价格敏感度分析 --------------------------
    with tab2:
        st.subheader("价格敏感度分析")

        # 价格敏感度计算函数
        def calculate_price_sensitivity(df, category_col='商品品类', price_col='实际售价', qty_col='销售数'):
            categories = df[category_col].unique()
            sensitivity_results = []

            for cat in categories:
                subset = df[df[category_col] == cat]
                if len(subset) < 10:
                    continue

                # 等频8分位数分组
                subset = subset.sort_values(price_col)
                try:
                    subset['price_bin'] = pd.qcut(subset[price_col], q=8, duplicates='drop')
                except:
                    continue

                bin_summary = subset.groupby('price_bin').agg({
                    price_col: 'mean',
                    qty_col: 'sum'
                }).reset_index()

                if len(bin_summary) < 3:
                    continue

                # 计算敏感度系数 S
                Qk = bin_summary[qty_col]
                Pk = bin_summary[price_col]
                Q_bar = Qk.mean()
                P_bar = Pk.mean()

                numerator = (Qk * ((Qk - Q_bar) / Q_bar)).sum()
                denominator = (Qk * ((Pk - P_bar) / P_bar)).sum()

                if denominator != 0:
                    S = numerator / denominator
                else:
                    S = np.nan

                # 判断敏感度等级
                if S < -0.5:
                    level = "强敏感"
                elif S < 0:
                    level = "弱敏感"
                else:
                    level = "不敏感"

                sensitivity_results.append({
                    '品类': cat,
                    '敏感度系数S': round(S, 4),
                    '敏感度等级': level
                })

            return pd.DataFrame(sensitivity_results)

        sensitivity_df = calculate_price_sensitivity(df)

        if not sensitivity_df.empty:
            # 显示敏感度结果
            st.dataframe(sensitivity_df, use_container_width=True)

            # 价格-销量拟合图生成
            st.subheader("价格-销量拟合分析")

            def generate_fit_plot(category):
                cat_data = df[df['商品品类'] == category].sort_values('实际售价').copy()

                # 等频分箱
                try:
                    cat_data['price_bin'] = pd.qcut(cat_data['实际售价'], q=8, duplicates='drop')
                except:
                    return None

                bin_summary = cat_data.groupby('price_bin').agg({
                    '实际售价': 'mean',
                    '销售数': 'sum'
                }).dropna().reset_index()

                if len(bin_summary) < 3:
                    return None

                P = bin_summary['实际售价'].values
                Q = bin_summary['销售数'].values

                # 拟合函数
                def linear_func(p, k, b):
                    return k * p + b

                def quadratic_func(p, a, b, c):
                    return a * p ** 2 + b * p + c

                # 生成图表
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(P, Q, color='#2E86AB', s=80, alpha=0.8, label='实际区间数据')

                # 线性拟合
                try:
                    linear_params, _ = curve_fit(linear_func, P, Q, maxfev=1000)
                    p_linear = np.linspace(P.min() * 0.9, P.max() * 1.1, 100)
                    ax.plot(p_linear, linear_func(p_linear, *linear_params),
                            color='#E74C3C', linewidth=2, linestyle='--',
                            label='线性拟合')
                except:
                    pass

                # 二次拟合
                try:
                    quadratic_params, _ = curve_fit(
                        quadratic_func, P, Q,
                        bounds=([-np.inf, -np.inf, 0], [0, np.inf, np.inf]),
                        maxfev=1000
                    )
                    p_quad = np.linspace(P.min() * 0.9, P.max() * 1.1, 100)
                    ax.plot(p_quad, quadratic_func(p_quad, *quadratic_params),
                            color='#27AE60', linewidth=2, label='二次拟合')

                    # 临界点
                    a, b_quad, c = quadratic_params
                    critical_price = -b_quad / (2 * a) if a != 0 else np.nan
                    if P.min() * 0.8 <= critical_price <= P.max() * 1.2:
                        ax.axvline(x=critical_price, color='#F39C12', linestyle='-.',
                                   label=f'临界点: {critical_price:.0f}元')
                except:
                    pass

                ax.set_title(f'{category} 价格-销量关系')
                ax.set_xlabel('实际售价（元）')
                ax.set_ylabel('销售数（件）')
                ax.legend()
                ax.grid(alpha=0.3)

                return fig

            # 选择品类查看拟合图
            selected_category = st.selectbox("选择品类查看价格-销量拟合图", sensitivity_df['品类'].unique())
            if selected_category:
                fig = generate_fit_plot(selected_category)
                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("该品类数据不足，无法生成拟合图")
        else:
            st.warning("没有足够数据计算价格敏感度")

    # -------------------------- 用户分层分析 --------------------------
    with tab3:
        st.subheader("用户分层价格偏好分析")

        def user_price_preference_analysis(df):
            # 计算全品类平均价格（加权）
            P_bar = (df['实际售价'] * df['销售数']).sum() / df['销售数'].sum()

            # 用户分层
            df['用户分层'] = df['客户年龄'].apply(
                lambda x: '青年' if x < 30 else '中年' if x < 50 else '老年'
            ) + '_' + df['客户性别'].apply(lambda x: '男' if x == '男' else '女')

            # 计算各分层指标
            layers = df['用户分层'].unique()
            results = []

            for layer in layers:
                subset = df[df['用户分层'] == layer]
                total_sales_qty = subset['销售数'].sum()

                if total_sales_qty == 0:
                    continue

                avg_price = (subset['实际售价'] * subset['销售数']).sum() / total_sales_qty

                # R1: 价格接受度
                R1 = avg_price / P_bar

                # R2: 价格集中度
                low_price_sales = subset[subset['实际售价'] < P_bar]['销售数'].sum()
                high_price_sales = subset[subset['实际售价'] >= P_bar]['销售数'].sum()

                R2_low = low_price_sales / total_sales_qty
                R2_high = high_price_sales / total_sales_qty

                # R3: 敏感倾向指数
                if R2_high != 0:
                    R3 = R2_low / R2_high
                else:
                    R3 = np.inf

                # 判断敏感度
                if R3 > 2.5:
                    sense_level = "强敏感"
                elif R3 > 1.5:
                    sense_level = "中敏感"
                else:
                    sense_level = "不敏感"

                results.append({
                    '用户分层': layer,
                    'R1': round(R1, 4),
                    'R2_low': round(R2_low, 4),
                    'R2_high': round(R2_high, 4),
                    'R3': round(R3, 4),
                    '敏感度等级': sense_level
                })

            return pd.DataFrame(results)

        user_pref_df = user_price_preference_analysis(df)

        if not user_pref_df.empty:
            # 显示用户分层结果
            st.dataframe(user_pref_df, use_container_width=True)

            # 可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # R1可视化
            user_pref_sorted = user_pref_df.sort_values('R1', ascending=False)
            bars1 = ax1.bar(user_pref_sorted['用户分层'], user_pref_sorted['R1'],
                            color='skyblue', alpha=0.8)
            ax1.axhline(y=1, color='red', linestyle='--', label='平均价格线')
            ax1.set_title('用户分层价格接受度 (R1)')
            ax1.set_ylabel('R1 (价格接受度)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend()

            # R3可视化
            user_pref_sorted = user_pref_df.sort_values('R3', ascending=False)
            colors = ['red' if x == '强敏感' else 'orange' if x == '中敏感' else 'green'
                      for x in user_pref_sorted['敏感度等级']]
            bars2 = ax2.bar(user_pref_sorted['用户分层'], user_pref_sorted['R3'],
                            color=colors, alpha=0.8)
            ax2.axhline(y=1.5, color='orange', linestyle='--', label='中敏感阈值')
            ax2.axhline(y=2.5, color='red', linestyle='--', label='强敏感阈值')
            ax2.set_title('用户分层价格敏感度 (R3)')
            ax2.set_ylabel('R3 (敏感倾向指数)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("用户分层分析数据不足")

    # -------------------------- 场景关联分析 --------------------------
    with tab4:
        st.subheader("场景关联分析")

        # 场景分析配置
        st.write("### 🎯 营销场景价格敏感度分析")

        # 创建场景定义
        scenarios = {
            '促销期': '分析促销活动期间的价格敏感度变化',
            '节假日': '节假日特殊消费场景分析',
            '新品上市': '新品推广期的价格策略效果',
            '常规期': '日常销售的价格敏感度基准'
        }

        selected_scenario = st.selectbox("选择分析场景", list(scenarios.keys()))
        st.info(f"当前场景：{selected_scenario} - {scenarios[selected_scenario]}")

        # 场景模拟分析
        col1, col2 = st.columns(2)

        with col1:
            st.write("#### 场景特征分析")

            # 基于场景的敏感度模拟
            scenario_multipliers = {
                '促销期': 1.8,  # 促销期敏感度提高
                '节假日': 1.3,  # 节假日适度提高
                '新品上市': 0.7,  # 新品期敏感度降低
                '常规期': 1.0  # 基准
            }

            multiplier = scenario_multipliers[selected_scenario]

            # 计算场景调整后的敏感度
            if 'sensitivity_df' in locals() and not sensitivity_df.empty:
                scenario_sensitivity = sensitivity_df.copy()
                scenario_sensitivity['场景调整系数'] = multiplier
                scenario_sensitivity['场景敏感度'] = scenario_sensitivity['敏感度系数S'] * multiplier

                # 重新分类场景敏感度
                def classify_scenario_sensitivity(s_value):
                    if s_value < -0.8:
                        return "极高敏感"
                    elif s_value < -0.4:
                        return "高敏感"
                    elif s_value < 0:
                        return "中等敏感"
                    else:
                        return "低敏感"

                scenario_sensitivity['场景敏感等级'] = scenario_sensitivity['场景敏感度'].apply(
                    classify_scenario_sensitivity
                )

                st.dataframe(
                    scenario_sensitivity[['品类', '敏感度系数S', '场景调整系数', '场景敏感度', '场景敏感等级']],
                    use_container_width=True)

        with col2:
            st.write("#### 场景策略建议")

            scenario_strategies = {
                '促销期': [
                    "🔥 重点品类加大折扣力度",
                    "🎯 设置阶梯式满减优惠",
                    "⚡ 限时抢购提升转化率",
                    "📱 推送个性化优惠券"
                ],
                '节假日': [
                    "🎁 节日主题营销活动",
                    "👨‍👩‍👧‍👦 家庭套装组合销售",
                    "🎄 限量版节日特别款",
                    "💝 礼品包装增值服务"
                ],
                '新品上市': [
                    "🌟 首发优惠吸引尝鲜",
                    "📢 KOL合作推广造势",
                    "🎪 线下体验活动联动",
                    "🔍 用户反馈快速迭代"
                ],
                '常规期': [
                    "📊 数据监控价格带表现",
                    "🔄 A/B测试优化定价",
                    "🎯 精准用户群体营销",
                    "📈 会员体系深度运营"
                ]
            }

            strategies = scenario_strategies[selected_scenario]
            for i, strategy in enumerate(strategies, 1):
                st.write(f"{i}. {strategy}")

        # 场景对比分析
        st.write("### 多场景对比分析")

        # 创建场景对比数据
        scenario_comparison = []
        for scenario, mult in scenario_multipliers.items():
            if 'sensitivity_df' in locals() and not sensitivity_df.empty:
                avg_sensitivity = sensitivity_df['敏感度系数S'].mean() * mult
                scenario_comparison.append({
                    '场景': scenario,
                    '调整系数': mult,
                    '平均敏感度': round(avg_sensitivity, 4),
                    '描述': scenarios[scenario]
                })

        scenario_df = pd.DataFrame(scenario_comparison)

        # 显示场景对比
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(scenario_df, use_container_width=True)

        with col2:
            # 场景对比可视化
            fig, ax = plt.subplots(figsize=(10, 6))
            scenarios_plot = scenario_df.sort_values('平均敏感度')
            colors = ['#ff9999' if x < 0 else '#66b3ff' for x in scenarios_plot['平均敏感度']]
            bars = ax.barh(scenarios_plot['场景'], scenarios_plot['平均敏感度'], color=colors, alpha=0.8)

            # 添加数值标签
            for bar, value in zip(bars, scenarios_plot['平均敏感度']):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{value:.3f}', va='center', fontsize=10)

            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='敏感度零线')
            ax.set_xlabel('平均价格敏感度')
            ax.set_title('不同营销场景的价格敏感度对比')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)


    # -------------------------- 综合策略生成 --------------------------
    with tab5:
        st.subheader("综合运营策略生成")

        # 确保有ABC分类和敏感度数据
        if 'category_summary' in locals() and 'sensitivity_df' in locals() and not sensitivity_df.empty:

            # 综合策略生成
            def generate_integrated_strategies(abc_df, sensitivity_df):
                strategies = []

                for _, abc_row in abc_df.iterrows():
                    cat = abc_row['商品品类']
                    abc_class = abc_row['ABC分类']

                    # 获取敏感度信息
                    sens_info = sensitivity_df[sensitivity_df['品类'] == cat]
                    if not sens_info.empty:
                        sens_level = sens_info['敏感度等级'].iloc[0]
                        S_value = sens_info['敏感度系数S'].iloc[0]
                    else:
                        sens_level = "未知"
                        S_value = 0

                    # 基于ABC分类和敏感度生成策略
                    if abc_class == 'A':
                        if sens_level == "强敏感":
                            strategy = "🔥 核心高敏感品类：重点监控价格，采用动态定价，推送高折扣券，设置价格预警"
                        elif sens_level == "弱敏感":
                            strategy = "⭐ 核心低敏感品类：适度溢价，强调品质和服务，推会员专属权益"
                        else:
                            strategy = "💎 核心品类：平衡定价，注重品牌建设，优化产品组合"

                    elif abc_class == 'B':
                        if sens_level == "强敏感":
                            strategy = "📊 重要高敏感品类：竞争性定价，参与平台促销，捆绑销售提升客单价"
                        else:
                            strategy = "📈 重要品类：适中定价，买赠活动，交叉销售推荐"

                    else:  # C类
                        if sens_level == "强敏感":
                            strategy = "🔄 一般高敏感品类：清理库存为主，大幅折扣，限时抢购"
                        else:
                            strategy = "📦 一般品类：维持现状，减少营销投入，自然销售"

                    strategies.append({
                        '品类': cat,
                        'ABC分类': abc_class,
                        '价格敏感度': sens_level,
                        '敏感度系数': S_value,
                        '推荐策略': strategy
                    })

                return pd.DataFrame(strategies)

            strategy_df = generate_integrated_strategies(category_summary, sensitivity_df)

            # 显示策略表
            st.dataframe(strategy_df, use_container_width=True)

            # 策略统计
            st.subheader("策略分布统计")
            col1, col2, col3 = st.columns(3)

            with col1:
                high_sens = len(strategy_df[strategy_df['价格敏感度'] == '强敏感'])
                st.metric("强敏感品类", high_sens)

            with col2:
                medium_sens = len(strategy_df[strategy_df['价格敏感度'] == '弱敏感'])
                st.metric("弱敏感品类", medium_sens)

            with col3:
                low_sens = len(strategy_df[strategy_df['价格敏感度'] == '不敏感'])
                st.metric("不敏感品类", low_sens)

            # 策略建议总结
            st.subheader("关键策略建议")

            # A类品类策略
            a_strategies = strategy_df[strategy_df['ABC分类'] == 'A']
            if not a_strategies.empty:
                st.write("### 🎯 A类核心品类策略")
                for _, row in a_strategies.iterrows():
                    st.write(f"- **{row['品类']}** ({row['价格敏感度']})：{row['推荐策略']}")

            # 高敏感品类策略
            high_sens_strategies = strategy_df[strategy_df['价格敏感度'] == '强敏感']
            if not high_sens_strategies.empty:
                st.write("### 💰 高敏感品类价格策略")
                for _, row in high_sens_strategies.iterrows():
                    st.write(f"- **{row['品类']}**：重点关注价格竞争力，建议定期调价")

        else:
            st.warning("请先完成ABC分类和价格敏感度分析")


# 注意：在实际使用时，确保这个函数被正确调用
# task4_operation_optimization()
def show_system_status():
    """系统状态页面"""
    st.markdown('<h2><i class="bi bi-gear icon-large text-secondary"></i> 系统状态</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="modern-card">
        <h4><i class="bi bi-check2-square icon-medium"></i> 任务完成状态</h4>
    </div>
    """, unsafe_allow_html=True)

    tasks = [
        ("数据预处理", st.session_state.task1_completed, "bi-file-earmark-excel"),
        ("多维特征分析", st.session_state.task2_completed, "bi-diagram-3"),
        ("销售预测", st.session_state.task3_completed, "bi-graph-up"),
        ("运营优化", st.session_state.task4_completed, "bi-lightbulb")
    ]

    for task_name, completed, icon_class in tasks:
        status_class = "status-completed" if completed else "status-pending"
        status_text = "已完成" if completed else "待完成"
        status_icon = "check-circle-fill" if completed else "clock-fill"

        st.markdown(f"""
            <div class="task-status {status_class}">
                <i class="bi {icon_class} icon-medium"></i>
                <i class="bi bi-{status_icon} icon-medium"></i>
                <span style="margin-left: 0.5rem;"><strong>{task_name}</strong> - {status_text}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="modern-card">
            <h4><i class="bi bi-database icon-medium"></i> 数据状态</h4>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.raw_data is not None:
            df = st.session_state.raw_data
            total_records = len(df)
            total_cols = len(df.columns)
            numeric_cols = len(st.session_state.column_types['numeric']) if st.session_state.column_types else 0
            category_cols = len(st.session_state.column_types['nominal']) + len(
                st.session_state.column_types['ordinal']) if st.session_state.column_types else 0

            # 数据状态指标卡片
            st.markdown(f"""
            <div class="metric-card">
                <i class="bi bi-file-earmark-text icon-large"></i>
                <h6>总记录数</h6>
                <h3>{total_records:,}条</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="modern-card text-center">
                <i class="bi bi-columns icon-large text-primary"></i>
                <h6>总字段数</h6>
                <h3>{total_cols}个</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="modern-card text-center">
                    <i class="bi bi-hash icon-large text-success"></i>
                    <h6>数值型字段</h6>
                    <h3>{numeric_cols}个</h3>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
                <div class="modern-card text-center">
                    <i class="bi bi-tags icon-large text-warning"></i>
                    <h6>分类型字段</h6>
                    <h3>{category_cols}个</h3>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                <h5><i class="bi bi-info-circle-fill"></i> 暂无数据</h5>
                <p>请先执行"数据预处理"任务</p>
            </div>
            """, unsafe_allow_html=True)

    # 系统操作区域
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="modern-card">
        <h4><i class="bi bi-tools icon-medium"></i> 系统操作</h4>
    </div>
    """, unsafe_allow_html=True)

    if st.button("重置系统", type="secondary", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()
        st.rerun()

    st.markdown("""
    <div class="info-box">
        <h6><i class="bi bi-info-circle"></i> 系统重置说明</h6>
        <p>重置系统将清除所有任务结果和缓存数据，系统将恢复到初始状态。此操作不可撤销。</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 主应用函数
# ============================================================================
# ============================================================================
# 主应用函数 - 修复路由问题
# ============================================================================
def main():
    """主应用函数"""
    # 大屏标题
    st.markdown("""
    <div class="dashboard-header animate-slide-in">
        <h1 class="dashboard-title">电商销售数据分析中心</h1>
        <p class="dashboard-subtitle">实时监控 · 智能分析 · 策略优化</p>
    </div>
    """, unsafe_allow_html=True)

    # 顶部导航菜单 - 透明按钮+Bootstrap Icons方案
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        if st.button("项目概览", key="nav_project_overview",
                    help="查看项目整体状态和概览信息",
                    use_container_width=True):
            st.session_state.current_task = "project_overview"

    with col2:
        if st.button("数据预处理", key="nav_task1",
                    help="数据清洗和预处理",
                    use_container_width=True):
            st.session_state.current_task = "task1"

    with col3:
        if st.button("多维分析", key="nav_task2",
                    help="多维度销售特征分析",
                    use_container_width=True):
            st.session_state.current_task = "task2"

    with col4:
        if st.button("销售预测", key="nav_task3",
                    help="智能销售预测分析",
                    use_container_width=True):
            st.session_state.current_task = "task3"

    with col5:
        if st.button("运营优化", key="nav_task4",
                    help="运营策略优化建议",
                    use_container_width=True):
            st.session_state.current_task = "task4"

    with col6:
        if st.button("系统状态", key="nav_system_status",
                    help="查看系统运行状态",
                    use_container_width=True):
            st.session_state.current_task = "system_status"

    # 添加全局样式覆盖所有按钮
    st.markdown("""
    <style>
    /* 覆盖所有按钮样式 */
    button {
        background: rgba(30,41,59,0.6) !important;
        border: 1px solid #475569 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        color: white !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
    }

    button:hover {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
        border-color: #00d4ff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,212,255,0.4) !important;
    }

    /* 确保主内容区域的按钮样式 */
    .main button {
        background: rgba(30,41,59,0.6) !important;
        border: 1px solid #475569 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        color: white !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
    }

    .main button:hover {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
        border-color: #00d4ff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0,212,255,0.4) !important;
    }

    /* 确保文字颜色正确 */
    button * {
        color: white !important;
    }

    /* 自定义Streamlit顶部栏样式 */
    .st-emotion-cache-1av3cm3 {
        background: rgba(10,14,26,0.95) !important;
        backdrop-filter: blur(10px) !important;
        border-bottom: 1px solid rgba(71,85,105,0.6) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    }

    .st-emotion-cache-1av3cm3 button {
        background: rgba(30,41,59,0.6) !important;
        border: 1px solid #475569 !important;
        color: white !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.3s ease !important;
    }

    .st-emotion-cache-1av3cm3 button:hover {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
        border-color: #00d4ff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(0,212,255,0.4) !important;
    }

    /* 隐藏右上角的部署按钮 */
    .st-emotion-cache-1kyxout {
        display: none !important;
    }

    /* 自定义右上角菜单图标 */
    .st-emotion-cache-17uys33 {
        color: white !important;
    }

    .st-emotion-cache-17uys33:hover {
        color: #00d4ff !important;
    }

    /* 隐藏右下角的报告问题按钮 */
    .st-emotion-cache-1yf3p4n {
        display: none !important;
    }

    /* 完全隐藏顶部栏的选项（取消注释即可使用） */
    /*
    .st-emotion-cache-1av3cm3 {
        display: none !important;
    }
    */

    /* 如果想隐藏侧边栏，取消下面的注释 */
    /*
    [data-testid="stSidebar"] {
        display: none !important;
    }
    */

    /* 修复所有字体颜色和对比度问题 */
    .main {
        color: white !important;
    }

    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: white !important;
    }

    .main p, .main span, .main div, .main label {
        color: #e2e8f0 !important;
    }

    .main strong {
        color: #ffffff !important;
    }

    /* 输入框文字颜色 */
    input, select, textarea {
        color: white !important;
        background-color: rgba(30,41,59,0.8) !important;
    }

    /* 选择框文字颜色 */
    .stSelectbox div[data-baseweb="select"] > div {
        color: white !important;
    }

    /* 文本输入框文字颜色 */
    .stTextInput input {
        color: white !important;
    }

    /* 数字输入框文字颜色 */
    .stNumberInput input {
        color: white !important;
    }

    /* 多选框样式 */
    .stMultiSelect div[data-baseweb="multiselect"] > div {
        color: white !important;
        background-color: rgba(30,41,59,0.8) !important;
    }

    /* 滑块样式 */
    .stSlider input[type="range"] {
        color: #00d4ff !important;
    }

    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #00d4ff, #7c3aed) !important;
    }

    /* 数据表格样式 */
    .dataframe {
        color: white !important;
        background: rgba(30,41,59,0.8) !important;
    }

    .dataframe th {
        background: rgba(51,65,85,0.8) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    .dataframe td {
        color: #e2e8f0 !important;
        border-color: rgba(71,85,105,0.5) !important;
    }

    .dataframe tbody tr:hover {
        background: rgba(71,85,105,0.3) !important;
    }

    /* 图表样式 - 使用深色主题 */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }

    /* matplotlib图表背景 */
    .matplotlib {
        background: rgba(30,41,59,0.8) !important;
        border-radius: 10px !important;
    }

    /* 设置matplotlib图表为深色主题 */
    </style>

    <script>
    // 修改Canvas表格样式的JavaScript
    function fixDataFrameStyles() {
        // 查找所有Canvas表格
        const canvases = document.querySelectorAll('[data-testid="data-grid-canvas"]');

        canvases.forEach(canvas => {
            // 修改Canvas背景
            canvas.style.background = 'rgba(15, 23, 42, 0.95)';
            canvas.style.borderRadius = '12px';

            // 修改Canvas内的表格样式
            const tables = canvas.querySelectorAll('table');
            tables.forEach(table => {
                table.style.color = '#ffffff';

                // 修改表头
                const headers = table.querySelectorAll('[role="columnheader"]');
                headers.forEach(header => {
                    header.style.color = '#00d4ff';
                    header.style.background = 'rgba(0,212,255,0.1)';
                    header.style.fontWeight = '600';
                });

                // 修改单元格
                const cells = table.querySelectorAll('[role="gridcell"]');
                cells.forEach(cell => {
                    cell.style.color = '#ffffff';
                    cell.style.background = 'transparent';
                    cell.style.fontWeight = '500';
                    cell.style.textShadow = '0 1px 2px rgba(0,0,0,0.5)';
                });
            });
        });

        // 修改Data Grid编辑器
        const dataEditors = document.querySelectorAll('.stDataFrameGlideDataEditor');
        dataEditors.forEach(editor => {
            editor.style.background = 'rgba(15, 23, 42, 0.95)';
            editor.style.border = '1px solid rgba(0,212,255,0.3)';
            editor.style.borderRadius = '12px';
            editor.style.boxShadow = '0 8px 25px rgba(0,212,255,0.15)';
        });
    }

    // 页面加载时执行
    document.addEventListener('DOMContentLoaded', fixDataFrameStyles);

    // Streamlit页面更新时执行
    const observer = new MutationObserver(function(mutations) {
        fixDataFrameStyles();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    // 定期检查并修复样式
    setInterval(fixDataFrameStyles, 1000);
    </script>
    """, unsafe_allow_html=True)

    # 设置matplotlib图表深色主题
    plt.style.use('dark_background')

    # 自定义图表颜色方案 - 数据大屏配色
    DASHBOARD_COLORS = {
        'primary': '#00d4ff',
        'secondary': '#7c3aed',
        'accent': '#ff006e',
        'success': '#10b981',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'info': '#06b6d4',
        'dark': '#1e293b',
        'light': '#e2e8f0',
        'grid': '#475569',
        'background': '#1e293b'
    }

    # 数据大屏色彩方案
    DASHBOARD_PALETTE = [
        '#00d4ff',  # 青色 - 主色
        '#7c3aed',  # 紫色 - 次色
        '#ff006e',  # 粉色 - 强调色
        '#10b981',  # 绿色 - 成功
        '#f59e0b',  # 橙色 - 警告
        '#06b6d4',  # 天蓝 - 信息
        '#f97316',  # 深橙
        '#8b5cf6',  # 浅紫
        '#ec4899',  # 粉红
        '#14b8a6'   # 青绿
    ]

    # 获取图表样式的函数
    def get_chart_style(style_type='bar'):
        """返回适合数据大屏的图表样式"""
        if style_type == 'bar':
            return {
                'facecolor': 'transparent',
                'edgecolor': DASHBOARD_COLORS['light'],
                'linewidth': 0,
                'alpha': 0.8
            }
        elif style_type == 'line':
            return {
                'color': DASHBOARD_COLORS['primary'],
                'linewidth': 3,
                'marker': 'o',
                'markersize': 6,
                'markerfacecolor': DASHBOARD_COLORS['accent'],
                'markeredgecolor': 'white',
                'markeredgewidth': 2
            }
        elif style_type == 'scatter':
            return {
                'color': DASHBOARD_COLORS['secondary'],
                'alpha': 0.7,
                'edgecolor': 'white',
                'linewidth': 1
            }
        else:
            return {}

    # 获取当前任务
    selected_key = st.session_state.get('current_task', 'project_overview')

    # 状态面板 - 实时数据展示
    completed_tasks = sum([st.session_state.task1_completed, st.session_state.task2_completed,
                          st.session_state.task3_completed, st.session_state.task4_completed])
    total_tasks = 4
    progress_percentage = (completed_tasks / total_tasks) * 100
    current_file = st.session_state.get('current_file', '未上传文件')

    st.markdown(f"""
    <div class="status-panel animate-slide-in">
        <div class="status-item">
            <div class="status-value">{completed_tasks}</div>
            <div class="status-label">已完成任务</div>
        </div>
        <div class="status-item">
            <div class="status-value">{total_tasks}</div>
            <div class="status-label">总任务数</div>
        </div>
        <div class="status-item">
            <div class="status-value">{progress_percentage:.0f}%</div>
            <div class="status-label">完成进度</div>
        </div>
        <div class="status-item">
            <div class="status-value">
                <div class="live-indicator">
                    <div class="live-dot"></div>
                    <span>实时</span>
                </div>
            </div>
            <div class="status-label">系统状态</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 页面路由
    if selected_key == "project_overview" or selected_key is None:
        show_project_overview()
    elif selected_key == "task1":
        task1_data_preprocessing()
    elif selected_key == "task2":
        enhanced_task2_multidimensional_analysis()
    elif selected_key == "task3":
        task3_sales_forecast()
    elif selected_key == "task4":
        task4_operation_optimization()
    elif selected_key == "system_status":
        show_system_status()

    # 页脚
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #e2e8f0; padding: 20px;'>
        <div class="modern-card" style="max-width: 600px; margin: 0 auto;">
            <i class="bi bi-code-square"></i>
            <strong>电商销售分析与策略优化系统</strong><br>
            <small>标准化输出 | 基于Bootstrap Icons和Sober UI构建</small>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# 运行应用
# ============================================================================
if __name__ == "__main__":
    main()
