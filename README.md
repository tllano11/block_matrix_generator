# Tabla de contenidos

- [Objetivos](#objetivos)
  - [Objetivo general](#objetivo-general)
  - [Objetivos específicos](objetivos-especificos)
- [Pseudocódigo](doc/pseudocodigo.md)

# Objetivos

## Objetivo general
Tratamiento de matrices de gran dimensión por bloques en GPU.

## Objetivos específicos
- Crear un generador de sistemas de ecuaciones lineales Ax=b, donde la matriz
A no pueda almacenarse completamente en la memoria global de la GPU.
- Utilizar Jacobi para la solución de ecuaciones lineales por bloques.
