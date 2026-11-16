// Terminal keys (raw mode), like Python's termios/tty/select code.
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <cstring>
#include "a3/render.hpp"

namespace a3 {
TerminalKeys::TerminalKeys() {
  static_assert(sizeof(termios) <= sizeof(saved_), "termios too big");
  if (!isatty(STDIN_FILENO)) return;
  termios t;
  if (tcgetattr(STDIN_FILENO, &t) != 0) return;
  std::memcpy(saved_, &t, sizeof(t));
  t.c_lflag &= ~(ICANON | ECHO);                   // like tty.setcbreak: keys without Enter
  t.c_cc[VMIN] = 0;
  t.c_cc[VTIME] = 0;
  tcsetattr(STDIN_FILENO, TCSANOW, &t);
  active_ = true;
}
TerminalKeys::~TerminalKeys() {
  if (active_) tcsetattr(STDIN_FILENO, TCSADRAIN, reinterpret_cast<termios*>(saved_));
}
bool TerminalKeys::read(char& c) {
  if (!active_) return false;
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(STDIN_FILENO, &fds);
  timeval tv{0, 0};
  if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) <= 0) return false;
  return ::read(STDIN_FILENO, &c, 1) == 1;
}
}  // namespace a3
