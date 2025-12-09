// 主题管理器 - 修改版
class ThemeManager {
    constructor() {
        this.themes = {
            light: { name: "浅色模式", bg: "#f8f9fa", text: "#2c3e50" },
            dark: { name: "深色模式", bg: "#1a1a1a", text: "#ffffff" },
            nature: { name: "自然模式", bg: "#f5f5dc", text: "#2c3e50" }
        };
        
        this.init();
    }
    
    init() {
        console.log("主题管理器初始化...");
        
        // 1. 恢复保存的主题
        const savedTheme = localStorage.getItem("theme") || "light";
        console.log("恢复主题:", savedTheme);
        this.applyTheme(savedTheme, true);  // true表示初始化时不显示通知
        
        // 2. 绑定主题按钮
        document.querySelectorAll(".theme-btn").forEach(btn => {
            btn.addEventListener("click", (e) => {
                const theme = e.currentTarget.getAttribute("data-theme");
                console.log("点击主题按钮:", theme);
                this.applyTheme(theme);
                this.updateActiveButton(theme);
                this.showNotification(`已切换到${this.themes[theme].name}`);
            });
        });
        
        // 3. 字体功能
        const fontSelector = document.getElementById("fontSelector");
        if (fontSelector) {
            // 恢复字体
            const savedFont = localStorage.getItem("font");
            if (savedFont) {
                fontSelector.value = savedFont;
                document.body.style.fontFamily = savedFont;
                console.log("恢复字体:", savedFont);
            }
            
            // 监听变化
            fontSelector.addEventListener("change", (e) => {
                const fontValue = e.target.value;
                document.body.style.fontFamily = fontValue;
                localStorage.setItem("font", fontValue);
                
                const fontName = e.target.options[e.target.selectedIndex].text;
                this.showNotification(`字体已更改为：${fontName}`);
                console.log("切换字体:", fontValue);
            });
        }
        
        // 4. 音频功能
        const audioSelector = document.getElementById("audioSelector");
        if (audioSelector) {
            const savedAudio = localStorage.getItem("audio-setting");
            if (savedAudio) audioSelector.value = savedAudio;
            
            audioSelector.addEventListener("change", (e) => {
                localStorage.setItem("audio-setting", e.target.value);
                this.showNotification(`音频设置已更新`);
            });
        }
        
        // 5. 更新按钮状态
        this.updateActiveButton(savedTheme);
        
        console.log("主题管理器初始化完成");
    }
    
    applyTheme(themeName, silent = false) {
        if (!this.themes[themeName]) {
            console.error("未知主题:", themeName);
            return;
        }
        
        const theme = this.themes[themeName];
        
        // 方法1：设置data-theme属性（让CSS变量生效）
        document.documentElement.setAttribute("data-theme", themeName);
        
        // 方法2：直接设置body样式（确保100%生效）
        document.body.style.backgroundColor = theme.bg;
        document.body.style.color = theme.text;
        
        // 方法3：保存到localStorage
        localStorage.setItem("theme", themeName);
        
        // 调试信息
        console.log(`应用主题: ${themeName}`);
        console.log(`设置背景色: ${theme.bg}`);
        console.log(`实际背景色: ${getComputedStyle(document.body).backgroundColor}`);
        
        if (!silent) {
            this.showNotification(`已切换到${theme.name}`);
        }
    }
    
    updateActiveButton(themeName) {
        document.querySelectorAll(".theme-btn").forEach(btn => {
            const btnTheme = btn.getAttribute("data-theme");
            if (btnTheme === themeName) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    }
    
    showNotification(message) {
        // 创建通知
        const notification = document.createElement("div");
        notification.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: var(--card-bg);
            color: var(--text-color);
            padding: 12px 20px;
            border-radius: var(--border-radius);
            box-shadow: var(--shadow-lg);
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 999;
            animation: slideInRight 0.3s ease;
            border-left: 4px solid var(--primary-color);
        `;
        
        // 添加动画
        const style = document.createElement("style");
        style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes fadeOut {
                to { opacity: 0; transform: translateY(-10px); }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        // 3秒后移除
        setTimeout(() => {
            notification.style.animation = "fadeOut 0.3s ease";
            setTimeout(() => {
                notification.remove();
                style.remove();
            }, 300);
        }, 3000);
    }
    
    // 调试方法
    debug() {
        console.log("=== 主题调试 ===");
        console.log("HTML data-theme:", document.documentElement.getAttribute("data-theme"));
        console.log("Body背景色:", getComputedStyle(document.body).backgroundColor);
        console.log("CSS变量--bg-color:", getComputedStyle(document.documentElement).getPropertyValue("--bg-color"));
        console.log("保存的主题:", localStorage.getItem("theme"));
        console.log("保存的字体:", localStorage.getItem("font"));
    }
}

// 初始化
document.addEventListener("DOMContentLoaded", () => {
    window.themeManager = new ThemeManager();
    
    // 确保Font Awesome加载
    if (!document.querySelector('link[href*="font-awesome"]')) {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css";
        document.head.appendChild(link);
    }
    
    // 添加调试快捷键 Ctrl+D
    document.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === "d") {
            e.preventDefault();
            if (window.themeManager) window.themeManager.debug();
        }
    });
});
