# О проекте

Выполнен в рамках хакатона летней школы РАИИ "Planning with Thunder".
Авторы: https://github.com/metra4ok https://github.com/VDStudy https://gitlab.com/ilya0_0 

# Planning with Thunder Workshop

В репозитории представлены подготовительные инструкции, сцены и docker-образы для участия в семинаре __`"Разработка алгоритма планирования движения с использованием платформы AgileX Robotics Thunder"`__ Летней школы Искусственного Интеллекта 2021 (RAAI Summer School 2021).

## Предварительная подготовка

Для участия в семинаре необходимо установить следующее ПО (рекомендуется использовать дистрибутив [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)):

1. [ROS](http://wiki.ros.org/ROS/Installation) (для Ubuntu 20.04 версии [Noetic](http://wiki.ros.org/noetic/Installation)).
2. Среду симуляции [CoppeliaSim Edu](https://www.coppeliarobotics.com/downloads);
3. [Docker](https://docs.docker.com/engine/install/ubuntu/);

### Установка ROS

Полные инструкции по установке представлены [здесь](http://wiki.ros.org/noetic/Installation/Ubuntu).

Для семинара необходимо установить сборку __`ros-*-desktop-full`__.

### Установка CoppeliaSim

Если архив с симулятором был скачан в директорию `~/Downloads`, то установка производится следующими коммандами:

```bash
cd ~/Downloads
tar -Jxf CoppeliaSim_Edu_V4_2_0_Ubuntu20_04.tar.xz --directory=/path/to/your/folder
```

> Для загрузки ROS плагина при запуске CoppeliaSim необходимо сначала запустить `roscore`:
> ```bash
> # terminal 1
> roscore
> # terminal 2
> cd /path/to/your/folder/CoppeliaSim_Edu_V4_2_0_Ubuntu20_04
> ./coppeliaSim.sh   
> ```
> В случае успешной загрузки плагинов вы увидите в терминале 2 сообщения:
> ```bash
> [CoppeliaSim:loadinfo]   plugin 'ROS': loading...
> [CoppeliaSim:loadinfo]   plugin 'ROS': load succeeded.
> ```

### Установка Docker

Рекомендуемый способ установки docker через добавление apt репозитория описан [здесь](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository).

Кроме того для упрощения дальнейшей работы рекомендуется выполнить [инструкции для работы non-root ползователей](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

## Запуск среды симуляции и готовых компонент

Для запуска тестового примера необходимо выполнить следующее:

1. Запустить ROS-мастера коммандой ```roscore``` в терминале 1.
2. Запустить симулятор `CoppeliaSim` в терминале 2, а затем и тестовую сцену.
3. Запустить docker-контейнер с предустановленными компонентами в терминале 3.
4. Запустить стандартную утилиту визуализации RViz в терминале 4.

Пдробнее эти инструкции описаны ниже.

### Запуск симулятора и тестовой сцены

В папке [./scenes](./scenes) находится подготовленная сцена для тестирования кода. После запуска через меню `File` ➡ `Open scene...` необходимо выбрать сцену [flat_polygon_hunter.ttt](./scenes/flat_polygon_hunter.ttt).

При первом запуске необходимо указать корректный путь к скрипту [hunter_control_script.lua](./scenes/lua/hunter_control_script.lua) в lua скрипте, ассоциированом с объектом `hull`.

После этого можно запускать сцену. Проверить правильность настройки можно выполнив следующую команду:
```bash
# показать список активных топиков
rostopic list
    ...
    /radar/points
    /odometry
    /cmd_vel
   ...
``` 
Среда симуляции начнет публиковать топики с данными одометрии и облаками точек.

![CoppeliaSim example](docs/coppelia.png)

### Запуск предустановленных компонент в docker-контейнере

Для работы предоставляется готовый docker-образ с предустановленными компонентами построения карты занятости, планирования траектории и др.

Готовый образ можно установить командой:
```bash
docker pull registry.gitlab.com/raai_planning_workshop/planning_with_thunder:latest
```

Для запуска и входа в контейнер предоставляются готовые скрипты в папке [./docker](./docker):
```bash
# запуск контейнера
./docker/start.sh

# вход в контейнер
./docker/into.sh 
```

При запуске создается директория `./planning_with_thunder/workspace/src`, которая затем маунтится в контейнер, и в которой можно размещать исходный код.

Также в контейнер маунтится папка [./launch](./launch), в которой расположены .launch-файлы для запуска предустановленых узлов.

Для запуска компонент внутри контейнера необходимо выполнить следующие команды:
```bash
source ~/catkin_ws/install/setup.bash
roslaunch ~/launch/main.launch
```

### Запуск среды визуализации RViz и управление платформой

В терминале выполнить команду:
```bash
# запуск RViz-а с пробросом конфигурационного файла
rosrun rviz rviz -d /your/path/here/planning_with_thunder/launch/rviz/thunder_sim.rviz

# запуск Rviz по умолчанию, конфигурационный файл прогружать нужно через меню
rviz
```

Предварительно настроенный конфигурационный файл: [thunder_sim.rviz](./launch/rviz/thunder_sim.rviz).

Кликая с помощью инструмента `2D Nav Goal` задаются ключевые точки маршрута, которые должен посетить робот. После выбора инструмента `Publish Point`, кликнув по карте маршрут отправляется на выполнение.

Пример прохождения маршрута роботом:

![RViz example](docs/rviz.png)

### Отладка в RQT

Фреймворк RQT предоставляет широкий набор инструментов для отладки системы в виде GUI плагинов.

Для запуска необходимо в отдельном терминале выполнить команду:
```bash
rqt
```

Все доступные инструменты сгруппированы в меню Plugins. Наиболее часто используемые плагины: Node Graph, Topic Monitor, Dynamic Reconfigure.

![RQT example](docs/rqt.png)

## Задание

Подробно задание на семинар описано в [отдельном файле](docs/problem.md).
