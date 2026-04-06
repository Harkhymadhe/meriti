.ONESHELL:

all: create run sync

create:
	@ruff check src/*
	@echo
	@script -q -c "wandb sweep sweep.yaml" sweep.log
	@echo

extract:
	@SWEEP_CONFIG=$$(./sweep.sh)
	@echo "$$SWEEP_CONFIG"
	@echo

run:
	@./data.sh
	@SWEEP_CONFIG=$$(./sweep.sh)
	@wandb agent "$$SWEEP_CONFIG"
	@echo

pause:
	@SWEEP_CONFIG=$$(./sweep.sh)
	@wandb sweep --pause "$$SWEEP_CONFIG"
	@echo

resume:
	@SWEEP_CONFIG=$$(./sweep.sh)
	@wandb sweep --resume "$$SWEEP_CONFIG"
	@echo

stop:
	@SWEEP_CONFIG=$$(./sweep.sh)
	@wandb sweep --stop "$$SWEEP_CONFIG"
	@echo

cancel:
	@SWEEP_CONFIG=$$(./sweep.sh)
	@wandb sweep --cancel "$$SWEEP_CONFIG"
	@echo

sync:
	@SUBDIRS=$$(find ./wandb -maxdepth 1 -type d | grep run)
	@echo $$SUBDIRS
	@for dir in $$SUBDIRS; do \
		wandb sync "$$dir"; \
		echo ; \
	done

profile:
	@./profile.sh
