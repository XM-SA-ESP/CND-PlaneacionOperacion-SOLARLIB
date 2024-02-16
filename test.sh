#!/bin/sh

if [ $1 != "docker" ]; then
    poetry run tox
    exit_code=$?
else
    docker-compose -f docker-compose.local.yml up --build
    exit_code=$?
fi

exit $exit_code