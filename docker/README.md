# Docker build steps
- 이 레포지토리를 클론받아 `docker` 폴더로 이동합니다. (혹은 이 폴더에 있는 README.md를 제외한 4개의 파일만 로컬로 복사해도 충분합니다.)
- `lmg_512_4.pt`를 로컬 폴더로 복사합니다.
    - 해당 파일이 없다면, 147.47.69.82의 `/home/jaechanlee/atpbind/lmg_512_4.pt`에 원본이 존재하니 이 원본을 가져와도 됩니다.
- `docker build -t <tag> .` 으로 빌드
- `docker run -v <directory with input file>:<directory in docker> <tag> python /script.py <input pdb file> <output csv file>` 으로 스크립트를 실행하면, 입력 pdb file에 대응되는 prediction이 output csv file에 기록됩니다. (residue index, prediction) 형식으로 기록됩니다.
  - 예시: `docker run -v $(pwd):/data/ lmg:latest python /script.py /data/pdb/3EPSA.pdb /data/out.csv`
  - https://github.com/dlwocks31/atpbind/tree/main/data/pdb 의 pdb file을 테스트 시 사용할 수 있습니다.