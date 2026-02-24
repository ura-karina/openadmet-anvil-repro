# Настройка прокси для расширения ChatGPT/Codex и Claude в Cursor

## Проблема

Расширения ChatGPT и Claude в Cursor не используют системные настройки прокси и получают ошибку 403 Forbidden от Cloudflare при попытке подключения к chatgpt.com/claude.ai.

## Решение

Создание wrapper-скриптов для бинарников `codex` и `claude`, которые устанавливают переменные окружения прокси перед запуском.

## Текущая конфигурация прокси

**Основной прокси-сервер**: vpn2 (91.84.100.35:3128)
- IP: `91.84.100.35`
- Порт: `3128`
- Аутентификация: Не требуется (IP whitelist)
- Доступ разрешён для: drug (172.16.17.200), Lobachevsky (212.41.30.60), VPN subnet (10.172.174.0/24)

**Резервный прокси-сервер**: vpn (185.157.214.65:3128) - **ВРЕМЕННО НЕДОСТУПЕН**
- Проблема с OpenVPN data channel, требуется связь с администраторами VPN

**SSH-конфигурация**: Все хосты используют ProxyJump через vpn2 для доступа к внутренним серверам.

**VPN Keepalive Service**: На vpn2 настроен systemd service `vpn-keepalive.service` который каждые 5 секунд пингует drug (172.16.17.200) через туннель, чтобы предотвратить inactivity timeout и перезапуски OpenVPN.

**⚠️ КРИТИЧЕСКИ ВАЖНО**: На vpn2 **ОБЯЗАТЕЛЬНО** должен быть настроен NAT masquerading для интерфейса tun0, иначе VPN туннель будет "засыпать" через ~60 секунд из-за отсутствия обратных пакетов от drug. См. раздел "Настройка NAT на vpn2" ниже.

---

## Шаги настройки

### Быстрое обслуживание после обновлений
- Если Cursor обновил расширения и прокси снова не работают: запустите `~/bin/fix-cursor-proxy` (он переобернёт новые бинарники `codex` и `claude` через общий скрипт `~/.cursor-server/extensions/proxy-wrapper.sh`).
- После запуска `fix-cursor-proxy` перезапустите расширения (или Cursor), чтобы подхватить обёртки.
- Терминальный `codex` (CLI) ставится через npm (`npm install -g @openai/codex@0.63.0`) и запускается из `~/bin/codex`, который сначала ищет npm-версию и только при отсутствии падает обратно на расширение.
- Если меняется адрес/порт прокси: отредактируйте `~/.cursor-server/extensions/proxy-wrapper.sh` (строки `HTTP_PROXY/HTTPS_PROXY`), сохраните и снова выполните `~/bin/fix-cursor-proxy`.

### 1. Найти директорию с бинарником codex расширения

```bash
ls -la ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/
```

Должен быть файл `codex` - это бинарник расширения ChatGPT.

### 2. Создать резервную копию оригинального бинарника

```bash
cd ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/
mv codex codex.orig
```

### 3. Создать wrapper-скрипт

**Используйте актуальный прокси-сервер: 91.84.100.35:3128**

```bash
cd ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/

cat > codex << 'EOF'
#!/bin/bash
# Wrapper for codex to use proxy

# Прокси-сервер vpn2
export HTTP_PROXY=http://91.84.100.35:3128
export HTTPS_PROXY=http://91.84.100.35:3128
export http_proxy=http://91.84.100.35:3128
export https_proxy=http://91.84.100.35:3128

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Execute the original codex binary
exec "${DIR}/codex.orig" "$@"
EOF

chmod +x codex
```

### 4. Проверить что wrapper создан корректно

```bash
ls -la ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/
```

Должно быть два файла:
- `codex` - wrapper-скрипт (маленький размер, ~400 байт)
- `codex.orig` - оригинальный бинарник (~40 MB)

### 5. Проверить содержимое wrapper

```bash
cat ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex
```

Убедитесь что прокси указан правильно.

### 6. Перезапустить расширение

Два варианта:

**Вариант А: Перезапустить процесс codex**
```bash
# Найти PID процесса
ps aux | grep "codex app-server" | grep $USER | grep -v grep

# Убить процесс (он автоматически перезапустится при следующем использовании)
kill <PID>
```

**Вариант Б: Полностью перезапустить Cursor**

Закройте и откройте Cursor заново.

---

## Проверка работы

### После перезапуска попробуйте использовать расширение ChatGPT

Откройте панель ChatGPT в Cursor и задайте любой вопрос.

### Проверить что процесс запущен с правильными переменными

```bash
# Найти PID процесса codex
PID=$(ps aux | grep "codex app-server" | grep $USER | grep -v grep | awk '{print $2}')

# Проверить переменные окружения (требуется sudo)
sudo cat /proc/$PID/environ | tr '\0' '\n' | grep -i proxy
```

Должны отобразиться переменные `HTTP_PROXY` и `HTTPS_PROXY` с вашим прокси.

---

## Дополнительная настройка (опционально)

### Глобальные переменные окружения для всех приложений

Добавьте в `~/.bashrc`:

```bash
# Proxy for Cursor/ChatGPT extension
export HTTP_PROXY=http://91.84.100.35:3128
export HTTPS_PROXY=http://91.84.100.35:3128
```

Затем выполните:
```bash
source ~/.bashrc
```

### Настройки прокси в Cursor UI

1. Откройте Cursor
2. Нажмите `Ctrl+,` (Settings)
3. Выберите вкладку **"Remote [SSH: ИМЯ_СЕРВЕРА]"** (если работаете по SSH)
4. Найдите "Http: Proxy"
5. Введите: `http://91.84.100.35:3128`
6. Убедитесь что "Http: Proxy Strict SSL" выключен (unchecked)

---

## Мониторинг (для отладки)

Создайте скрипт для мониторинга процессов codex:

```bash
cat > /tmp/monitor_codex.sh << 'EOF'
#!/bin/bash
echo "Мониторинг процессов codex и их сетевых соединений..."
echo "Нажмите Ctrl+C для остановки"
echo ""

while true; do
    clear
    echo "=== Процессы codex ==="
    ps aux | grep -E "(codex|chatgpt)" | grep $USER | grep -v grep | grep -v monitor
    echo ""
    echo "=== Переменные окружения ==="
    for pid in $(ps aux | grep "codex app-server" | grep $USER | grep -v grep | awk '{print $2}'); do
        echo "PID: $pid"
        sudo cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep -i proxy || echo "  (нет переменных прокси)"
        echo ""
    done
    sleep 3
done
EOF

chmod +x /tmp/monitor_codex.sh
```

Запуск:
```bash
/tmp/monitor_codex.sh
```

---

## Wrapper для терминальных версий codex и claude

### Codex wrapper (~/bin/codex)

```bash
mkdir -p ~/bin
cat > ~/bin/codex << 'EOF'
#!/bin/bash
# Wrapper for codex-cli to use VPN proxy
export HTTP_PROXY=http://91.84.100.35:3128
export HTTPS_PROXY=http://91.84.100.35:3128

# Automatically find the latest OpenAI ChatGPT extension version
CODEX_PATH=$(find ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex 2>/dev/null | sort -V | tail -1)

if [ -z "$CODEX_PATH" ]; then
    echo "Error: codex binary not found in Cursor extensions" >&2
    exit 1
fi

exec "$CODEX_PATH" "$@"
EOF

chmod +x ~/bin/codex
```

### Claude wrapper (~/bin/claude)

```bash
cat > ~/bin/claude << 'EOF'
#!/bin/bash
# Wrapper for Claude CLI to use VPN proxy
export HTTP_PROXY=http://91.84.100.35:3128
export HTTPS_PROXY=http://91.84.100.35:3128
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
exec ~/.nvm/versions/node/v22.20.0/bin/claude "$@"
EOF

chmod +x ~/bin/claude
```

Убедитесь что `~/bin` в вашем `$PATH`:
```bash
export PATH="$HOME/bin:$PATH"
```

**Эти врапперы уже настроены на Lobachevsky. Для drug скопируйте файлы с Desktop:**
```bash
scp ~/Desktop/codex_wrapper_drug.sh drug:~/bin/codex
scp ~/Desktop/claude_wrapper_drug.sh drug:~/bin/claude
ssh drug 'chmod +x ~/bin/codex ~/bin/claude'
```

---

## Восстановление в случае проблем

Если что-то пошло не так, восстановите оригинальный бинарник:

```bash
cd ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/
mv codex.orig codex
```

---

## Параметры прокси

**Основной прокси** (vpn2):
- **Host**: `91.84.100.35`
- **Port**: `3128`
- **Протокол**: HTTP (не HTTPS, не SOCKS)
- **Аутентификация**: Не требуется (используется IP whitelist)
- **Доступ**: drug, Lobachevsky, VPN subnet

**Резервный прокси** (vpn) - ВРЕМЕННО НЕДОСТУПЕН:
- **Host**: `185.157.214.65`
- **Port**: `3128`
- **Проблема**: OpenVPN data channel не работает, требуется диагностика

Если прокси требует аутентификации (в будущем):
```bash
export HTTP_PROXY=http://username:password@91.84.100.35:3128
export HTTPS_PROXY=http://username:password@91.84.100.35:3128
```

---

## Troubleshooting

### Расширение все равно не работает

1. Убедитесь что wrapper имеет права на выполнение:
   ```bash
   chmod +x ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex
   ```

2. Проверьте что `codex.orig` существует и это бинарный файл:
   ```bash
   file ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex.orig
   ```

3. Убедитесь что прокси-сервер доступен:
   ```bash
   curl -x http://91.84.100.35:3128 https://chatgpt.com
   ```

   Если не работает, проверьте доступность с Lobachevsky:
   ```bash
   ssh Lobachevsky 'curl -x http://91.84.100.35:3128 -s https://ifconfig.me'
   ```
   Должен вернуть IP vpn2: `91.84.100.35`

4. Проверьте логи Cursor:
   ```bash
   tail -f ~/.cursor-server/data/logs/*/exthost*.log
   ```

### Процесс codex не запускается

Проверьте синтаксис wrapper-скрипта:
```bash
bash -n ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/codex
```

Если есть ошибки - исправьте скрипт.

### После обновления расширения wrapper пропал

При обновлении расширения ChatGPT файлы могут перезаписаться. Повторите шаги 2-3 после каждого обновления расширения.

Можно автоматизировать через скрипт:
```bash
cat > ~/bin/fix_cursor_proxy.sh << 'EOF'
#!/bin/bash
CODEX_DIR=$(find ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64 -type d 2>/dev/null | head -1)

if [ -z "$CODEX_DIR" ]; then
    echo "Error: ChatGPT extension not found"
    exit 1
fi

cd "$CODEX_DIR"

if [ ! -f "codex.orig" ]; then
    mv codex codex.orig
fi

cat > codex << 'WRAPPER'
#!/bin/bash
export HTTP_PROXY=http://91.84.100.35:3128
export HTTPS_PROXY=http://91.84.100.35:3128
export http_proxy=http://91.84.100.35:3128
export https_proxy=http://91.84.100.35:3128
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
exec "${DIR}/codex.orig" "$@"
WRAPPER

chmod +x codex
echo "Proxy wrapper установлен в $CODEX_DIR"
EOF

chmod +x ~/bin/fix_cursor_proxy.sh
```

Запускайте после каждого обновления расширения:
```bash
~/bin/fix_cursor_proxy.sh
```

---

## VPN Keepalive Service (на vpn2)

Для предотвращения засыпания VPN туннеля из-за inactivity timeout на сервере, настроен systemd service который постоянно поддерживает активность туннеля.

### Скрипт keepalive (/root/vpn-keepalive.sh)

```bash
#!/bin/bash
# VPN tunnel keepalive - prevents inactivity timeout
# Ping every 5 seconds to keep tunnel active
while true; do
    ping -c 1 -W 2 172.16.17.200 > /dev/null 2>&1
    sleep 5
done
```

### Systemd service (/etc/systemd/system/vpn-keepalive.service)

```ini
[Unit]
Description=VPN Tunnel Keepalive Service
After=openvpn-custom.service
Requires=openvpn-custom.service

[Service]
Type=simple
ExecStart=/root/vpn-keepalive.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Управление service

```bash
# Проверить статус
systemctl status vpn-keepalive

# Перезапустить
systemctl restart vpn-keepalive

# Просмотр логов
journalctl -u vpn-keepalive -f

# Отключить (если нужно)
systemctl stop vpn-keepalive
systemctl disable vpn-keepalive
```

### Проверка работы

Service должен быть в состоянии `active (running)` и пинговать drug каждые 5 секунд:

```bash
# На vpn2
systemctl status vpn-keepalive
journalctl -u vpn-keepalive --since "1 minute ago"
```

---

## VPN Watchdog Service (на vpn2)

Watchdog автоматически перезапускает OpenVPN если туннель падает.

### Скрипт watchdog (/root/vpn-watchdog.sh)

```bash
#!/bin/bash
# VPN tunnel watchdog - automatically restarts OpenVPN if tunnel is down

DRUG_IP="172.16.17.200"
CHECK_INTERVAL=30
PING_TIMEOUT=5

while true; do
    # Try to ping drug through VPN tunnel
    if ! ping -c 1 -W "$PING_TIMEOUT" "$DRUG_IP" > /dev/null 2>&1; then
        echo "[$(date)] Tunnel down, restarting OpenVPN..."
        systemctl restart openvpn-custom
        sleep 5
    fi
    sleep "$CHECK_INTERVAL"
done
```

### Systemd service (/etc/systemd/system/vpn-watchdog.service)

```ini
[Unit]
Description=VPN Tunnel Watchdog Service
After=openvpn-custom.service
Requires=openvpn-custom.service

[Service]
Type=simple
ExecStart=/root/vpn-watchdog.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Управление watchdog

```bash
# Проверить статус
systemctl status vpn-watchdog

# Включить/выключить
systemctl enable vpn-watchdog
systemctl disable vpn-watchdog

# Запустить/остановить
systemctl start vpn-watchdog
systemctl stop vpn-watchdog
```

---

## Настройка NAT на vpn2 (⚠️ КРИТИЧЕСКИ ВАЖНО!)

**Почему это важно**: Без NAT masquerading, пакеты от vpn2 к drug уходят с внешним IP (91.84.100.35), drug пытается отправить ответы через свой default gateway (не через VPN), ответные пакеты не возвращаются через туннель, и VPN сервер считает туннель неактивным и закрывает его через ~60 секунд.

### Включить IP forwarding

```bash
ssh vpn2 'echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward'
```

Для автоматического применения после перезагрузки добавьте в `/etc/sysctl.conf`:
```bash
net.ipv4.ip_forward = 1
```

### Настроить NAT masquerading

```bash
ssh vpn2 'sudo iptables -t nat -A POSTROUTING -o tun0 -j MASQUERADE'
```

### Проверить правила

```bash
ssh vpn2 'sudo iptables -t nat -L POSTROUTING -n -v'
```

Должно быть правило:
```
MASQUERADE  all  --  *      tun0    0.0.0.0/0            0.0.0.0/0
```

### Сохранить правила (чтобы пережили перезагрузку)

```bash
# Установить iptables-persistent
ssh vpn2 'DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent'

# Сохранить текущие правила
ssh vpn2 'netfilter-persistent save'
```

---

## Проверка работы на обоих серверах

### Drug (172.16.17.200)

```bash
# Проверить codex CLI
ssh drug '~/bin/codex --version'

# Проверить claude CLI
ssh drug '~/bin/claude --version'

# Проверить proxy
ssh drug 'curl -x http://91.84.100.35:3128 -s https://ifconfig.me'

# Проверить wrapper для расширения
ssh drug 'ls -la ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/ | grep codex'
```

Ожидаемые результаты:
- codex: версия 0.53.0
- claude: версия 2.0.14
- proxy: должен вернуть `91.84.100.35`
- wrapper: должны быть файлы `codex` (~300 байт) и `codex.orig` (~49 MB)

### Lobachevsky (212.41.30.60)

```bash
# Проверить codex CLI
ssh Lobachevsky '~/bin/codex --version'

# Проверить claude CLI
ssh Lobachevsky '~/bin/claude --version'

# Проверить proxy
ssh Lobachevsky 'curl -x http://91.84.100.35:3128 -s https://ifconfig.me'

# Проверить wrapper для расширения
ssh Lobachevsky 'ls -la ~/.cursor-server/extensions/openai.chatgpt-*/bin/linux-x86_64/ | grep codex'
```

Ожидаемые результаты:
- codex: версия 0.53.0
- claude: версия 2.0.34
- proxy: должен вернуть `91.84.100.35`
- wrapper: должны быть файлы `codex` (~300 байт) и `codex.orig` (~49 MB)

