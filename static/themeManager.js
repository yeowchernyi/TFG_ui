// 主题管理器
class ThemeManager {
    constructor() {
        this.themes = {
            light: { name: "浅色模式", icon: "fa-sun" },
            dark: { name: "深色模式", icon: "fa-moon" },
            nature: { name: "自然模式", icon: "fa-leaf" }
        };
        
        this.init();
    }
    
    init() {
        // 从localStorage读取保存的主题
        const savedTheme = localStorage.getItem("theme") || "light";
        this.applyTheme(savedTheme);
        
        // 绑定主题切换按钮
        document.querySelectorAll(".theme-btn").forEach(btn => {
            btn.addEventListener("click", (e) => {
                const theme = e.currentTarget.getAttribute("data-theme");
                this.applyTheme(theme);
                this.updateActiveButton(theme);
                this.showNotification(`已切换到${this.themes[theme].name}`);
            });
        });
        
        // 字体选择
        const fontSelector = document.getElementById("fontSelector");
        if (fontSelector) {
            fontSelector.addEventListener("change", (e) => {
                document.body.style.fontFamily = e.target.value;
                localStorage.setItem("font-family", e.target.value);
                this.showNotification(`字体已更改为：${e.target.options[e.target.selectedIndex].text}`);
            });
            
            // 恢复保存的字体
            const savedFont = localStorage.getItem("font-family");
            if (savedFont) {
                fontSelector.value = savedFont;
                document.body.style.fontFamily = savedFont;
            }
        }
        
        // 音频设置
        const audioSelector = document.getElementById("audioSelector");
        if (audioSelector) {
            audioSelector.addEventListener("change", (e) => {
                localStorage.setItem("audio-setting", e.target.value);
                this.showNotification(`音频设置已更新`);
            });
            
            // 恢复音频设置
            const savedAudio = localStorage.getItem("audio-setting");
            if (savedAudio) {
                audioSelector.value = savedAudio;
            }
        }
        
        this.updateActiveButton(savedTheme);
    }
    
    applyTheme(themeName) {
        if (!this.themes[themeName]) return;
        
        // 设置data-theme属性
        document.documentElement.setAttribute("data-theme", themeName);
        
        // 保存到localStorage
        localStorage.setItem("theme", themeName);
    }
    
    updateActiveButton(themeName) {
        document.querySelectorAll(".theme-btn").forEach(btn => {
            if (btn.getAttribute("data-theme") === themeName) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    }
    
    showNotification(message, type = "info") {
        // 移除现有的通知
        const existing = document.querySelector(".theme-notification");
        if (existing) existing.remove();
        
        // 创建通知元素
        const notification = document.createElement("div");
        notification.className = "theme-notification";
        notification.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
        `;
        
        // 样式
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
        
        // 3秒后自动消失
        setTimeout(() => {
            notification.style.animation = "fadeOut 0.3s ease";
            setTimeout(() => {
                notification.remove();
                style.remove();
            }, 300);
        }, 3000);
    }
}

// 初始化主题管理器
document.addEventListener("DOMContentLoaded", () => {
    window.themeManager = new ThemeManager();
    
    // 添加Font Awesome图标库
    if (!document.querySelector("link[href*='font-awesome']")) {
        const faLink = document.createElement("link");
        faLink.rel = "stylesheet";
        faLink.href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css";
        document.head.appendChild(faLink);
    }
});
