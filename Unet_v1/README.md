# CGIAR-Malaria

- `scripts`: carpeta que contiene códigos para pre/post procesamiento de las imágenes de drones. Eg. generación de patches.
- `scr`: carpeta que contiene los códigos fuente para el entrenamiento de la red.
- `sshAWS`: carpeta que contiene los archivos necesarios para la conexión al servidor de AWS y entrenar la red. 

## Antes de clonar

Primero actualizar:

```
sudo apt update
```

- Ubuntu 18.04:

En caso no tenga `pip3`:

```shell
sudo apt install python3-pip
```

Instalar `pipenv`:

```shell
pip3 install pipenv
```

En caso no tenga permisos:

```shell
pip3 install --user pipenv
```

Para AWS colocar lo siguiente:

```shell
sudo -H pip3 install -U pipenv
```

## Luego de clonar

Dentro del directorio clonado, ejecutar el siguiente comando para sincronizar las dependencias:

```shell
pipenv sync
```

Para activar el entorno:

```shell
pipenv shell
```

Para ejecutar el entorno de `jupyter`:

```shell
pipenv run jupyter notebook
```

Para salir del entorno de `pipenv`:

```shell
exit
```

