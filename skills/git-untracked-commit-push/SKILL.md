---
name: git-untracked-commit-push
description: Leer y versionar solo archivos untracked en repositorios Git, redactar un commit message con comentarios basados en los cambios y hacer push a la rama actual. Usar cuando el usuario pida publicar archivos nuevos sin incluir modificaciones tracked existentes.
---

# Git Untracked Commit Push

## Objetivo

Aplicar un flujo seguro para:
- detectar archivos untracked,
- revisar su contenido para describir cambios,
- crear un commit con asunto + comentarios,
- hacer push al remoto de la rama activa.

## Workflow

1. Detectar archivos objetivo:
- Ejecutar `git ls-files --others --exclude-standard`.
- Si no hay archivos, informar y terminar sin commit.

2. Revisar cambios antes de stage:
- Leer cada untracked (texto) para entender proposito.
- Si hay binarios, resumir por nombre/extension/tamano.
- Generar un resumen corto de cambios para el mensaje de commit.

3. Redactar commit message basado en cambios:
- Asunto en imperativo, <=72 caracteres.
- Cuerpo con comentarios concretos de impacto.
- Usar formato recomendado de `references/playbook.md`.

4. Stage estricto solo de untracked:
- Usar `git add -- <archivo1> <archivo2> ...` con la lista detectada.
- No incluir tracked modificados ni borrados.
- Validar con `git diff --cached --name-status`.

5. Commit y push:
- Commit con `-m "<asunto>" -m "<comentarios>"`.
- Detectar rama con `git branch --show-current`.
- Push con `git push origin <rama>`.
- Si la rama no tiene upstream, usar `git push -u origin <rama>`.

6. Reportar resultado:
- Mostrar hash de commit, rama y remoto.
- Listar archivos incluidos en el commit.

## Guardrails

- No hacer `git add .` en este flujo.
- No mezclar en el commit cambios tracked no solicitados.
- Si hay conflicto entre lo pedido y el estado del repo, confirmar con el usuario antes de continuar.
- Si `push` falla por autenticacion/permisos, reportar error exacto y no reintentar en bucle.

## Referencia

Usar `references/playbook.md` como patron de comandos y plantilla de mensaje.
