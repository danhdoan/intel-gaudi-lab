import http from 'k6/http';
import { sleep } from 'k6';


export let options = {
  vus: 20, 
  duration: '30s',
};

export default function () {
    let payload = JSON.stringify({
        prompt: "What is artificial intelligence?",
        max_new_tokens: 50
    });

    let params = {
        headers: { 'Content-Type': 'application/json' },
    };

    let res = http.post('http://localhost:8080/generate', payload, params);

    if (res.status !== 200) {
        console.error(`❌ Request failed with status ${res.status}`);
    }

    sleep(1);
}
