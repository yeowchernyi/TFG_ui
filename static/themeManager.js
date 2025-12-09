// 主题管理器 - 直接生效版
class ThemeManager {
    constructor() {
        this.themes = ['light', 'dark', 'nature'];
        this.init();
    }
    
    init() {
        // 设置默认主题
        const savedTheme = localStorage.getItem('theme') || 'light';
        this.setTheme(savedTheme);
        
        // 绑定按钮点击
        document.querySelectorAll('.theme-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const theme = e.target.getAttribute('data-theme');
                this.setTheme(theme);
                alert(`已切换到${theme}主题`);
            });
        });
        
        // 字体切换
        const fontSelect = document.getElementById('fontSelector');
        if (fontSelect) {
            fontSelect.addEventListener('change', (e) => {
                document.body.style.fontFamily = e.target.value;
                localStorage.setItem('font', e.target.value);
            });
            
            // 恢复字体
            const savedFont = localStorage.getItem('font');
            if (savedFont) {
                fontSelect.value = savedFont;
                document.body.style.fontFamily = savedFont;
            }
        }
    }
    
    setTheme(theme) {
        // 1. 设置html属性
        document.documentElement.setAttribute('data-theme', theme);
        
        // 2. 保存
        localStorage.setItem('theme', theme);
        
        // 3. 直接修改body背景色（确保生效）
        const colors = {
            light: '#f8f9fa',
            dark: '#1a1a1a',
            nature: '#f5f5dc'
        };
        document.body.style.backgroundColor = colors[theme] || colors.light;
    }
}

// 启动
document.addEventListener('DOMContentLoaded', () => {
    window.themeManager = new ThemeManager();
    
    // 添加图标库
    if (!document.querySelector('link[href*="font-awesome"]')) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css';
        document.head.appendChild(link);
    }
});
