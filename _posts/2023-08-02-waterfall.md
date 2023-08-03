---
layout: post
title: "async.waterfall"
---

# async.waterfall 이란?

js에서 비동기 코드를 처리하는데 사용되는 함수 중 하나. 여러 개의 비동기 작업을 순ㅊ나적으로 실행하는데 사용된다. 각 작업은 이전 작업의 결과에 의존하며, 각 작업이 완료되면 다음 작업을 호출한다.

## 예시

```javascript
async.waterfall(
  [
    function (callback) {
      //첫 번째 비동기 작업 수행
      //결과를 콜백으로 전달 : callback(err,result1)
    },
    function (result1, callback) {
      //두 번째 비동기 작업 수행
      //결과를 콜백으로 전달 : callback(err,result2)
    },
    //추가적인 비동기 작업들
  ],
  function (err, finalResult) {
    //모든 작업이 완료되면 실행되는 콜백ㄱ 함수
    //err는 에러 객체이며, finalResult은 마지막 작업의 결과입니다.
  }
);
```

첫 번째 함수는 항상 콜백 함수를 첫 번째 인자로 받는다. 각 함수는 비동기 작업을 ㅜ행하고, 결과나 에러를 다음 함수에 전달하기 위해 콜백함수를 호출한다. 마지막 인자로 주어지는 콜백 함수는 모든 작업이 완료되거나, 에러가 발생되었을 때 호출되며, 최종 결과를 받는다.
