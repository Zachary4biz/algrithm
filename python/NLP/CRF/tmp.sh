#!/usr/bin/env bash
kafka-console-consumer \
--consumer-property group.id="consumer_test_worker01" \
--property print.timestamp=true \
--bootstrap-server "xly-server" \
--topic "xly-topic"
