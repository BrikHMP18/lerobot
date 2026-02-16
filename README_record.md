# SO-ARM100: Recoleccion de Data con LeRobot

Guia rapida para dejar todo listo y grabar dataset con:

- Follower: `/dev/ttyACM0`
- Leader: `/dev/ttyACM1`
- Camaras usadas para dataset: `top` y `wrist`

## 1. Activar entorno

```bash
cd ~/NONHUMAN/lerobot
conda activate lerobot
```

## 2. Permisos serial (Linux)

Agrega tu usuario al grupo `dialout`:

```bash
sudo usermod -aG dialout $USER
newgrp dialout
```

Verifica:

```bash
groups
ls -l /dev/ttyACM0 /dev/ttyACM1
```

Si hace falta, permiso temporal:

```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1
```

## 3. Ver puertos seriales (leader/follower)

```bash
lerobot-find-port
```

Confirma:

- Follower -> `/dev/ttyACM0`
- Leader -> `/dev/ttyACM1`

## 4. (Solo si hace falta) Setear IDs de servos

Si tu follower o leader no responde durante calibracion, rehace IDs.

Follower:

```bash
lerobot-setup-motors \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0
```

Leader:

```bash
lerobot-setup-motors \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM1
```

## 5. Calibrar follower y leader

Follower:

```bash
lerobot-calibrate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so100_follower_main
```

Leader:

```bash
lerobot-calibrate \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=so100_leader_main
```

Archivos de calibracion esperados:

- `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so100_follower_main.json`
- `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so100_leader_main.json`

## 6. Ver camaras

```bash
lerobot-find-cameras opencv
```

Define el mapeo que usaras en todos los comandos (teleoperate, record, replay, eval):

```bash
# Ajusta estos valores segun tu maquina
TOP_CAM=/dev/video2
WRIST_CAM=/dev/video0
```

Notas:

- Si en tu equipo la wrist aparece en otro indice, cambia solo `WRIST_CAM`.
- Para entrenar `pi05`, `smolvla` y `act`, manten los mismos nombres de camara (`top`, `wrist`) y la misma configuracion durante recoleccion y evaluacion.

## 7. Teleoperar (prueba completa robot + camara)

```bash
lerobot-teleoperate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so100_follower_main \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=so100_leader_main \
  --display_data=true
```

## 8. Grabar dataset (local, sin subir al Hub)

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so100_follower_main \
  --robot.cameras="{ top: {type: opencv, index_or_path: ${TOP_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --teleop.type=so100_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=so100_leader_main \
  --display_data=true \
  --dataset.repo_id=autobrik/so100_test \
  --dataset.num_episodes=5 \
  --dataset.single_task="pick and place" \
  --dataset.push_to_hub=false
```

## 9. Troubleshooting rapido

Error tipico:

`Full found motor list (id: model_number): {}`

Significa que hay energia pero no comunicacion de datos en ese bus.

Checklist:

- Confirmar puerto con `lerobot-find-port`.
- Rehacer IDs en ese brazo con `lerobot-setup-motors`.
- Revisar cable USB de datos.
- Revisar cable 3 pines y orientacion.
- Revisar cadena daisy-chain desde `shoulder_pan` (ID 1) hasta `gripper` (ID 6).
- Revisar fuente de poder.
