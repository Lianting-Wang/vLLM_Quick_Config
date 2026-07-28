const form = document.querySelector('#login-form');
const password = document.querySelector('#password');
const error = document.querySelector('#login-error');

(async () => {
  try {
    const response = await fetch('/api/session');
    const data = await response.json();
    if (!data.authentication_required || data.authenticated) location.replace('/');
  } catch (_) {}
})();

form.addEventListener('submit', async event => {
  event.preventDefault();
  error.classList.add('hidden');
  try {
    const response = await fetch('/api/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({password: password.value})
    });
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 429) throw new Error('尝试次数过多，请一分钟后再试。');
      throw new Error('密码错误。');
    }
    location.replace('/');
  } catch (exception) {
    error.textContent = exception.message;
    error.classList.remove('hidden');
    password.select();
  }
});
