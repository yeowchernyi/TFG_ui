// ====== 主题管理器 - 三保险强制生效版 ======
(function() {
    console.log('🎨 主题管理器启动');
    
    // 主题定义
    const themes = {
        light: { bg: '#f8f9fa', text: '#2c3e50', name: '浅色' },
        dark: { bg: '#1a1a1a', text: '#ffffff', name: '深色' },
        nature: { bg: '#f5f5dc', text: '#2c3e50', name: '自然' }
    };
    
    // 三重保险应用主题
    function applyTheme(theme) {
        console.log('🔄 切换主题:', theme);
        const colors = themes[theme] || themes.light;
        
        // 保险1: 直接设置body样式（最高优先级）
        document.body.style.backgroundColor = colors.bg;
        document.body.style.color = colors.text;
        
        // 保险2: 设置CSS变量
        document.documentElement.style.setProperty('--bg-color', colors.bg);
        document.documentElement.style.setProperty('--text-color', colors.text);
        
        // 保险3: 设置HTML属性
        document.documentElement.setAttribute('data-theme', theme);
        
        // 保存到本地存储
        localStorage.setItem('theme', theme);
        
        // 更新按钮状态
        updateButtons(theme);
        
        showNotice(`已切换到${colors.name}模式`);
    }
    
    // 更新按钮状态
    function updateButtons(theme) {
        document.querySelectorAll('.theme-btn').forEach(btn => {
            const isActive = btn.getAttribute('data-theme') === theme;
            btn.classList.toggle('active', isActive);
        });
    }
    
    // 字体切换
    function initFontSelector() {
        const selector = document.getElementById('fontSelector');
        if (!selector) return;
        
        // 恢复保存的字体
        const savedFont = localStorage.getItem('font');
        if (savedFont) {
            selector.value = savedFont;
            document.body.style.fontFamily = savedFont;
        }
        
        // 监听变化
        selector.addEventListener('change', function() {
            const font = this.value;
            document.body.style.fontFamily = font;
            localStorage.setItem('font', font);
            showNotice(`字体已更改`);
        });
    }
    
    // 绑定按钮事件
    function bindThemeButtons() {
        document.querySelectorAll('.theme-btn').forEach(btn => {
            // 移除旧事件，绑定新事件
            const newBtn = btn.cloneNode(true);
            btn.parentNode.replaceChild(newBtn, btn);
            
            newBtn.addEventListener('click', function() {
                const theme = this.getAttribute('data-theme');
                applyTheme(theme);
            });
        });
    }
    
    // 显示通知
    function showNotice(text) {
        const notice = document.createElement('div');
        notice.textContent = text;
        notice.style.cssText = `
            position: fixed; top: 80px; right: 20px;
            background: #4A90E2; color: white; padding: 10px 20px;
            border-radius: 6px; z-index: 9999; font-size: 14px;
            animation: fadeIn 0.3s;
        `;
        document.body.appendChild(notice);
        setTimeout(() => notice.remove(), 2000);
    }
    
    // 初始化
    function init() {
        console.log('🚀 初始化主题系统...');
        
        // 应用保存的主题
        const savedTheme = localStorage.getItem('theme') || 'light';
        applyTheme(savedTheme);
        
        // 绑定事件
        bindThemeButtons();
        initFontSelector();
        
        console.log('✅ 主题系统初始化完成');
        
        // 调试函数
        window.debugTheme = function() {
            console.log('=== 主题调试 ===');
            console.log('当前主题:', localStorage.getItem('theme'));
            console.log('Body背景色:', getComputedStyle(document.body).backgroundColor);
            console.log('Body文字色:', getComputedStyle(document.body).color);
            console.log('data-theme属性:', document.documentElement.getAttribute('data-theme'));
        };
        
        window.resetTheme = function() {
            localStorage.clear();
            location.reload();
        };
    }
    
    // 启动
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
