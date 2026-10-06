# Yandex DataSphere: доступы и гайд (Wunder)

## Идентификаторы
- Community ID: `bt1dv4jmd0u81i806t74`
- Project ID:   `bt1dqf9jg0tuhggcrj4k`
- Консоль проекта: https://datasphere.yandex.cloud/communities/bt1dv4jmd0u81i806t74/projects/bt1dqf9jg0tuhggcrj4k?tab=project-jobs
- OAuth-токен (выдан 2026-10-04): `<SET_YOUR_DATASPHERE_OAUTH_TOKEN>`
  ВНИМАНИЕ: токены, выданные после 2026-06-01, yc для обмена на IAM НЕ принимает
  ("OAuth token ... issued after '2026-06-01', is not supported for IAM token exchange").
  Рабочий путь — вход через браузер на Mac (`yc init`) и ключ сервисного аккаунта для beleriand (ниже).

## Где что установлено
- Mac: `~/yandex-cloud/bin/yc` (1.39.0)
- beleriand: `~/yandex-cloud/bin/yc` (1.39.0), `~/ds-venv/bin/datasphere` (pip install datasphere)
- beleriand не может логиниться через браузер → использовать authorized key сервисного аккаунта:
  `~/.config/yandex-cloud/sa-datasphere-key.json` (создаётся с Mac, см. ниже)

## Разовая настройка
1. Mac: `~/yandex-cloud/bin/yc init` → вход через браузер, выбрать облако/папку проекта.
2. Mac: создать сервисный аккаунт и ключ:
   yc iam service-account create --name ds-wunder
   yc resource-manager folder add-access-binding <FOLDER_ID> --role editor --subject serviceAccount:<SA_ID>
   (+ в DataSphere: добавить ds-wunder участником проекта/сообщества с ролью Editor)
   yc iam key create --service-account-name ds-wunder -o sa-datasphere-key.json
   scp sa-datasphere-key.json beleriand:.config/yandex-cloud/
3. beleriand:
   yc config profile create ds-sa
   yc config set service-account-key ~/.config/yandex-cloud/sa-datasphere-key.json
   yc config set folder-id <FOLDER_ID>

## Запуск задания
config.yaml:
    name: wunder-exp
    cmd: python train.py --data ${DATA} --out ${OUT}
    env:
      python: auto        # переносит код и pip-зависимости
    inputs:
      - data.tar: DATA
    outputs:
      - out.tar: OUT
    cloud-instance-type: g1.1   # 1×V100; g2.1 = 1×A100; c1.4 = CPU
Команда:
    ~/ds-venv/bin/datasphere project job execute -p bt1dqf9jg0tuhggcrj4k -c config.yaml
Фоном (не ждать): `... job execute ... &` или `datasphere project job list -p <id>`, `job attach --id <job_id>`.
Большие входы лучше один раз положить в S3/датасет проекта, а не передавать в каждом задании.

## РАБОЧАЯ СХЕМА (проверено 2026-10-04)
- Организация ysda: создавать облако НЕ нужно (Permission denied — это нормально).
- Mac: `yc init` выполнен (вход через браузер, аккаунт aje707ehgf8rcunfh6ss). Проверка: `~/yandex-cloud/bin/yc iam create-token`.
- Токен на beleriand: launchd-агент `~/Library/LaunchAgents/com.wunder.ycpush.plist` каждые 3 ч запускает
  `~/Documents/wunder/datasphere/push_token.sh` → кладёт IAM-токен (живёт 12 ч) в beleriand:~/.config/yc_iam_token. Лог /tmp/ycpush.log.
  Mac должен быть включён хотя бы раз в 12 ч; если токен протух — запустить push_token.sh вручную.
- beleriand: обёртка `~/ds` (= datasphere с YC_IAM_TOKEN из файла):
    ~/ds project job execute -p bt1dqf9jg0tuhggcrj4k -c config.yaml
- Главный скрипт задания обязан содержать `if __name__ == '__main__':`.
- g1.1 = Tesla V100 32 GB, 8 vCPU, 94 GB RAM, 34 GB диска; старт ~1 мин; Python 3.12; torch не предустановлен (ставится через env/requirements).
- Пробное задание: ~/dsjobs/probe (job bt1l2phc87irihst7olg).

## Доступные машины (проверено 2026-10-04)
| тип | GPU | vCPU | RAM |
|---|---|---|---|
| g2.1 | A100-SXM4 80 GB (самая сильная) | 28 | 116 GB |
| g1.1 | V100 32 GB | 8 | 94 GB |
| gt4i.1 | L4 24 GB | 8 | 31 GB |
| gt4.1 | T4 16 GB | 4 | 15 GB |
Недоступны для сообщества: g1.2, g2.2, g2.mig, gh100.1, g3.1 (мульти-GPU и H100 нет).
