# Playbook: Untracked -> Commit -> Push

## 1) Detectar untracked

```powershell
git ls-files --others --exclude-standard
```

## 2) Stage solo untracked

```powershell
git add -- <archivo1> <archivo2> <archivo3>
git diff --cached --name-status
```

## 3) Mensaje de commit recomendado

Asunto:
- Imperativo y corto (<=72 chars)
- Ejemplo: `Agregar skill para commit/push de archivos untracked`

Cuerpo (comentarios):
- `Cambios:`
- `- <archivo o grupo>: <que agrega o corrige>`
- `- <archivo o grupo>: <impacto>`
- `Motivo: <por que se agrega>`

Ejemplo:

```text
Agregar skill para flujo untracked commit/push

Cambios:
- skills/git-untracked-commit-push/SKILL.md: define workflow para leer untracked, crear commit message y push.
- skills/git-untracked-commit-push/references/playbook.md: agrega comandos y plantilla de mensaje.
Motivo: estandarizar el flujo de publicacion de archivos nuevos sin mezclar cambios tracked.
```

## 4) Commit y push

```powershell
git commit -m "<asunto>" -m "<cuerpo>"
$branch = git branch --show-current
git push origin $branch
```

Fallback si no hay upstream:

```powershell
git push -u origin $branch
```
